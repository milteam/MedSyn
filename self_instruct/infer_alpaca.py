import os
import sys
import json
import random
from tqdm import tqdm

import fire
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from peft import PeftConfig, PeftModel, set_peft_model_state_dict

from utils.utils import read_jsonl


def generate_prompt(record, templates):
    print('templates', len(templates), templates)
    print('record', record)
    if "input" in record and record["input"]:
        template = random.choice(templates["prompts_input"])
        print('template', template)
        return template.format(instruction=record["instruction"], input=record["input"])
    template = random.choice(templates["prompts_no_input"])
    return template.format(instruction=record["instruction"])


def generate_answers(
    model_name: str,
    template_path: str,
    input_path: str,
    output_path: str,
    batch_size: int = 1
):
    with open(input_path) as f:
        records = json.load(f)

    assert batch_size == 1, "Batch inference is not yet supported"
    with open(template_path) as r:
        templates = json.load(r)
  
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(model_name)

    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(
        model,
        model_name,
        torch_dtype=torch.float16
    )

    resume_from_checkpoint = "models/ru_llama_7b_lora/checkpoint-230"
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

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with open(output_path, "w") as w:
        for record in tqdm(records):
            text = generate_prompt(record, templates)
            data = tokenizer(text, return_tensors="pt")
            data = {k: v.to(model.device) for k, v in data.items() if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                output_ids = model.generate(
                    **data,
                    generation_config=generation_config
                )[0]
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            answer = output.split(templates["output_separator"])[-1].strip()
            print("-----------Input text for Alpaca----")
            print(text)
            print("-----------Answer of Alpaca----")
            print(answer)
            record["answer"] = answer
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(generate_answers)
