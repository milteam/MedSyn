import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForTokenClassification
from transformers import AutoTokenizer

from utils.dataset import InstructDataset, ChatDataset
from utils.utils import fix_tokenizer, fix_model, set_random_seed


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        template_path: str = "templates/alpaca.json",  # The prompt template to use, will default to alpaca.
        source_field: str = "input",  # in the template
        target_field: str = "output",  # in the template
        model_type: str = "causal",  # "causal" or "seq2seq"
        mode: str = "instruct",  # "instruct" or "chat"
        max_source_tokens_count: int = 256,
        max_target_tokens_count: int = 512,
        max_tokens_count: int = 2000,
        train_sample_rate: float = 1.0,
        val_sample_rate: float = 1.0,
        only_target_loss: bool = True,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        val_set_size: int = 2000,
        warmup_steps: int = 100,
        seed: int = 42,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
):
    set_random_seed(seed)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        gradient_accumulation_steps = batch_size // micro_batch_size
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"max_source_tokens_count: {max_source_tokens_count}\n"
            f"max_target_tokens_count: {max_target_tokens_count}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {template_path}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    print(f'gradient_accumulation_steps: {gradient_accumulation_steps}\n')

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # LlamaTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer = fix_tokenizer(tokenizer)
    tokenizer.save_pretrained(output_dir)

    # LlamaForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = fix_model(model, tokenizer)

    # Default model generation params
    model.config.num_beams = 5
    if mode == "instruct":
        max_tokens_count = max_target_tokens_count + max_source_tokens_count + 1
    model.config.max_length = max_tokens_count if model_type == "causal" else max_target_tokens_count

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if mode == "instruct":
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=seed
            )
            train_data = InstructDataset(
                train_val["train"],
                tokenizer,
                max_source_tokens_count=max_source_tokens_count,
                max_target_tokens_count=max_target_tokens_count,
                sample_rate=train_sample_rate,
                input_type=model_type,
                template_path=template_path,
                target_field=target_field,
                source_field=source_field,
                only_target_loss=only_target_loss
            )
            val_data = InstructDataset(
                train_val["test"],
                tokenizer,
                max_source_tokens_count=max_source_tokens_count,
                max_target_tokens_count=max_target_tokens_count,
                sample_rate=val_sample_rate,
                input_type=model_type,
                template_path=template_path,
                target_field=target_field,
                source_field=source_field,
                only_target_loss=only_target_loss
            )
        else:
            train_data = InstructDataset(
                data["train"],
                tokenizer,
                max_source_tokens_count=max_source_tokens_count,
                max_target_tokens_count=max_target_tokens_count,
                sample_rate=train_sample_rate,
                input_type=model_type,
                template_path=template_path,
                target_field=target_field,
                source_field=source_field,
                only_target_loss=only_target_loss
            )
            val_data = None
    elif mode == "chat":
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=seed
            )
            train_data = ChatDataset(
                train_val["train"],
                tokenizer,
                max_tokens_count=max_tokens_count,
                sample_rate=train_sample_rate,
                template_path=template_path,
                only_target_loss=only_target_loss
            )

            val_data = ChatDataset(
                train_val["test"],
                tokenizer,
                max_tokens_count=max_tokens_count,
                sample_rate=train_sample_rate,
                template_path=template_path,
                only_target_loss=only_target_loss
            )
        else:
            train_data = ChatDataset(
                data["train"],
                tokenizer,
                max_tokens_count=max_tokens_count,
                sample_rate=train_sample_rate,
                template_path=template_path,
                only_target_loss=only_target_loss
            )
            val_data = None
    else:
        assert False

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if "seq2seq" in model_type:
        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=10 if val_set_size > 0 else None,
            save_steps=10,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=data_collator,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    lora_ckpt_path = '/'.join(checkpoint_name.split('/')[:2])
    gen_config_filepath = f'{lora_ckpt_path}/generation_config.json'
    print('generation config file path', gen_config_filepath)
    if os.path.exists(gen_config_filepath):
        print('Copy `generation_config.json` to the output_dir')
        shutil.copyfile(gen_config_filepath, f'{output_dir}/generation_config.json')
    else:
        print(f'`generation_config.json` is not copied to the `{output_dir}` folder. '
              f'You may need to create `generation_config.json` manually')

if __name__ == "__main__":
    fire.Fire(train)
