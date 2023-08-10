from transformers import AutoTokenizer
import transformers
import torch

model = "huggyllama/llama-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'Напиши текст анамнеза, составленного врачом по итогам приема пациента. Пациент - незамужний курящий мужчина, '
    'который жалуется на носовые кровотечения, припухлость на шее, увеличенный лимфатический узел в области нижней '
    'челюсти, язвы в ротовой полости.',
    do_sample=True,
    top_k=40,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
