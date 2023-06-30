import json
import random

import click

from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}
### Response:"""

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    print(prompt)
    print('===============')
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()



# def generate_prompt_ru(info: DiseaseInfo):
#     return [{"role": "user", "content": f'''
#             Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный отчет.
# Представь, что у пациента есть слеующие хронические заболевания: {", ".join(info.preconditions)}.
# В действительности диагноз пациента - {info.name}. Но на момент написания анамнеза это еще неизвестно.
# Пациент жалуется на {", ".join(info.symptoms)}. Реалистично придумай все недостающие данные для отчета.'''},
#             ]


def generate_prompt_en(data):
    actual_desease = data["disease"][0]["Name"]
    symptoms = data["symptoms"]
    if len(symptoms) > 3:
        symptoms = random.sample(symptoms, random.randint(1,4))
    return actual_desease, symptoms, f'''
    Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections.        
  In fact, the patient has {actual_desease}, but the doctor does not know about it.
  The patient complains of {", ".join(symptoms)}.
  '''

@click.command()
@click.option('--samples', '-s', help='JSON files witj deseases', required=True)
def main(samples):
    res = []
    with open(samples, encoding="utf8") as f:
        data = json.load(f)
        for idx, desease in data.items():
            actual_desease, symptoms, prompt = generate_prompt_en(desease)
            result = evaluate(prompt)
            print(result)
            print("========================")
            res.append({'desease': actual_desease, 'symptoms': symptoms, 'description': result})

if __name__ == '__main__':
    main()