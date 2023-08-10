#--output_path 'output/ru_baseline_llama-v2-7b_saiga-v2_generated.jsonl' \

docker run --gpus '"device=5"' --shm-size 64g -p 7860:7860 \
  -v "${HOME}"/.cache:/root/.cache \
  -v "${HOME}"/MedTexts/alpaca-lora/output:/workspace/output --rm alpaca-lora \
  python3.10 infer_alpaca.py \
    --model_name 'models/eng_tloen-alpaca_7b' \
    --template_path 'templates/alpaca.json' \
    --input_path 'test.jsonl' \
    --output_path 'output/test.jsonl'
