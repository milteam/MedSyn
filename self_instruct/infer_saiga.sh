docker run --gpus '"device=5"' --shm-size 64g -p 7860:7860 \
-v "${HOME}"/.cache:/root/.cache \
-v "${HOME}"/MedTexts/alpaca-lora/output:/workspace/output --rm alpaca-lora \
python3.10 infer_saiga.py \
  --model_name 'models/ru_saiga-v2_7b' \
  --template_path 'templates/saiga_v2.json' \
  --input_path 'test.jsonl' \
  --output_path 'output/test.jsonl' \
