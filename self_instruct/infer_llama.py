import json
import os
import sys

import fire
import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter
from utils.utils import read_jsonl


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

def main(
        base_model: str,
        input_path: str,
        output_path: str,
        template_name: str = "alpaca",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(template_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    generation_config = GenerationConfig.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    records = read_jsonl(input_path)
    with open(output_path, "w") as w:
        for record in tqdm(records):
            prompt = prompter.generate_prompt(record['instruction'], record['input'])
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            # Without streaming
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    #max_new_tokens=20,
                    do_sample=True,
                    #top_k=40,
                    max_length=200,
                    #top_p=0.9,
                    #num_beams=1,
                )

            s = generation_output.sequences[0]
            output = tokenizer.decode(s, skip_special_tokens=True)
            answer = prompter.get_response(output)
            print(prompt)
            print(answer)
            record["answer"] = answer
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
