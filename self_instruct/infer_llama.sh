docker run --gpus '"device=5"' --shm-size 64g -p 7860:7860 \
-v "${HOME}"/.cache:/root/.cache \
-v "${HOME}"/MedTexts/alpaca-lora/output:/workspace/output --rm alpaca-lora \
python3.10 infer_llama.py \
  --base_model 'models/meta-llama-v2-7b' \
  --template_name 'ru_alpaca' \
  --input_path 'test.jsonl' \
  --output_path 'output/ru_baseline_llama-v2-7b.jsonl'
