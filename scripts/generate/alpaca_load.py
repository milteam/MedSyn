from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
)
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")
