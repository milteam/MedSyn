#--output_path 'output/ru_baseline_llama-v2-7b_saiga-v2_generated.jsonl' \

docker run--gpus '"device=4,5"' --shm-size 64g -p 7860:7860 \
  -v "${HOME}"/cache:/root/.cache \
  -v "${HOME}"/SberMedText/self_instruct/data/:/workspace/data \
  -v "${HOME}"/SberMedText/self_instruct/models/:/workspace/models \
  -v "${HOME}"/SberMedText/self_instruct/:/workspace/code \
  -v "${HOME}"/SberMedText/self_instruct/output:/workspace/output --rm alpaca-lora \
  python3.10 code/infer_alpaca.py \
    --model_name 'IlyaGusev/llama_7b_ru_turbo_alpaca_lora' \
    --template_path 'templates/alpaca.json' \
    --input_path 'data/testset.json' \
    --output_path 'output/testset.json'
