docker run --gpus '"device=5"' --shm-size 64g -p 7860:7860 \
  -v "${HOME}"/.cache:/root/.cache \
  -v "${HOME}"/MedTexts/alpaca-lora/output:/workspace/output --rm alpaca-lora \
  python3.10 hf_generation.py
