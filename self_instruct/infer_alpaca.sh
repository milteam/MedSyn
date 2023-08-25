docker run --gpus '"device=4,5"' --shm-size 64g -p 7860:7860 \
  -v "${HOME}"/.cache:/root/.cache \
  -v "${HOME}"/SberMedText/self_instruct/data/:/workspace/data \
  -v "${HOME}"/SberMedText/self_instruct/models/:/workspace/models \
  -v "${HOME}"/SberMedText/self_instruct/output:/workspace/output --rm alpaca-lora-test \
  python3.10 infer_alpaca.py \
    --model_name 'models/ru_llama_7b_lora' \
    --template_path 'templates/ru_alpaca.json' \
    --input_path 'data/testset.json' \
    --output_path 'output/testset.json'
