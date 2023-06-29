


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
#     if input:
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
#
# ### Instruction:
# {instruction}
#
# ### Input:
# {input}
#
# ### Response:"""
#     else:
#         return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
#
# ### Instruction:
# {instruction}
    return instruction

### Response:"""

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
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
        print(output)#"Response:", output.split("### Response:")[1].strip())

evaluate('''
Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections.
The patient has chronic diseases: gastritis, tonsillitis.
In fact, the patient has an exacerbation of chronic gastritis, but the doctor does not know about it.
''')