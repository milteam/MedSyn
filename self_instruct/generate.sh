docker run --gpus '"device=5"' --shm-size 64g -p 7860:7860 \
  -v "${HOME}"/.cache:/root/.cache --rm alpaca-lora \
  python3.10 generate.py \
    --base_model 'models/decapoda-llama-7b' \
    --lora_weights 'models/eng_tloen-alpaca_7b'
