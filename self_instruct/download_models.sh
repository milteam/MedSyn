MODEL_DIR='models/decapoda-llama-7b'
BASE_MODEL='decapoda-research/llama-7b-hf'
LORA_DIR='models/ru_turbo_alpaca_7b'
LORA_WEIGHTS='IlyaGusev/llama_7b_ru_turbo_alpaca_lora'
HF_TOKEN=''

docker run --name alpaca -v "${HOME}"/.cache:/root/.cache \
  -v "${HOME}"/MedTexts/alpaca-lora/"${MODEL_DIR}":/workspace/"${MODEL_DIR}" \
  -v "${HOME}"/MedTexts/alpaca-lora/"${LORA_DIR}":/workspace/"${LORA_DIR}" \
  --env MODEL_DIR=$MODEL_DIR \
  --env BASE_MODEL=$BASE_MODEL \
  --env LORA_DIR=$LORA_DIR \
  --env LORA_WEIGHTS=$LORA_WEIGHTS \
  --env HF_TOKEN=HF_TOKEN \

  --rm alpaca-lora \
  python3.10 download_models.py
