from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="medalpaca/medalpaca-lora-7b-8bit", tokenizer="medalpaca/medalpaca-lora-7b-8bit", config="medalpaca/medalpaca-lora-7b-8bit/adapter_config.json")
question = "What are the symptoms of diabetes?"
context = "Diabetes is a metabolic disease that causes high blood sugar. The symptoms include increased thirst, frequent urination, and unexplained weight loss."
answer = qa_pipeline({"question": question, "context": context})
print(answer)