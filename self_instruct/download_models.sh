MODEL_DIR='models/meta-llama-v2-7b'
BASE_MODEL='meta-llama/Llama-2-7b-hf'
LORA_DIR='models/ru_saiga-v2_7b'
LORA_WEIGHTS='IlyaGusev/saiga2_7b_lora'

docker run --name alpaca -v "${HOME}"/.cache:/root/.cache \
  -v "${HOME}"/MedTexts/alpaca-lora/"${MODEL_DIR}":/workspace/"${MODEL_DIR}" \
  -v "${HOME}"/MedTexts/alpaca-lora/"${LORA_DIR}":/workspace/"${LORA_DIR}" \
  --env MODEL_DIR=$MODEL_DIR \
  --env BASE_MODEL=$BASE_MODEL \
  --env LORA_DIR=$LORA_DIR \
  --env LORA_WEIGHTS=$LORA_WEIGHTS \
  --rm alpaca-lora \
  python3.10 download_models.py
