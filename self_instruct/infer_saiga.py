import copy
import json

import torch
from tqdm import tqdm

import fire
from transformers import AutoTokenizer, GenerationConfig, AutoConfig, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

from utils.chat import Conversation
from utils.utils import read_jsonl, gen_batch


def generate(model, tokenizer, prompts, generation_config):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )
    outputs = []
    for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
        sample_output_ids = sample_output_ids[len(sample_input_ids):]
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        sample_output = sample_output.replace("</s>", "").strip()
        outputs.append(sample_output)
    return outputs


def generate_answers(
        model_name: str,
        template_path: str,
        input_path: str,
        output_path: str,
        batch_size: int = 1,
        torch_dtype: str = None
):
    generation_config = GenerationConfig.from_pretrained(model_name)
    config = PeftConfig.from_pretrained(model_name)
    base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)

    if torch_dtype is not None:
        torch_dtype = getattr(torch, torch_dtype)
    else:
        torch_dtype = base_model_config.torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        #base_model_name_or_path,
        torch_dtype=torch_dtype,
        load_in_8bit=False,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(
        model,
        model_name,
        torch_dtype=torch_dtype
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if batch_size > 1:
        assert tokenizer.padding_side == "left", "Batched inference for right padding side is impossible"
    records = read_jsonl(input_path)

    default_conversation = Conversation.from_template(template_path)
    with open(output_path, "w") as w:
        for batch in tqdm(gen_batch(records, batch_size)):
            prompts = []
            for record in batch:
                conversation = copy.deepcopy(default_conversation)
                user_message = record["instruction"]
                if "input" in record and record["input"]:
                    user_message += "\nДано: " + record["input"]
                conversation.add_user_message(user_message)
                prompt = conversation.get_prompt(tokenizer)
                prompts.append(prompt)
            outputs = generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                generation_config=generation_config
            )
            for record, prompt, output in zip(batch, prompts, outputs):
                print(prompt)
                print(output)
                record["instruction"] = record["instruction"].encode("utf-8").decode("utf-8", "ignore")
                if "input" in record and record["input"]:
                    record["input"] = record["input"].encode("utf-8").decode("utf-8", "ignore")
                record["answer"] = output.encode("utf-8").decode("utf-8", "ignore")
                w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(generate_answers)
