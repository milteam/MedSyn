## self_instruct finetuning

### Setup

Build the container image:
```bash
docker build -t alpaca-lora .
```

### Training (`finetune.py`)
Run the container with `finetune.sh` script or use following command:
```bash
docker run --gpus '"device=3,4"' --shm-size 64g -p 7860:7860 --name alpaca \
  -v "${HOME}"/.cache:/root/.cache \
  --rm alpaca-lora \
  python3.10 finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --resume_from_checkpoint='yahma/alpaca-cleaned'
```

### Inference (`infer_alpaca.py`)
To generate samples run
```bash
docker run --gpus '"device=4,5"' --shm-size 64g -p 7860:7860 \
  -v "${HOME}"/.cache:/root/.cache \
  -v "${HOME}"/SberMedText/self_instruct/data/:/workspace/data \
  -v "${HOME}"/SberMedText/self_instruct/models/:/workspace/models \
  -v "${HOME}"/SberMedText/self_instruct/output:/workspace/output --rm alpaca-lora-test \
  python3.10 infer_alpaca.py \
    --model_name 'models/ru_llama_7b_lora' \      # path to a local lora weights
    --template_path 'templates/ru_alpaca.json' \
    --input_path 'data/testset.json' \            # test instructions
    --output_path 'output/testset.json'

```
Define `generation_config.json` in the `model_name` folder if needed
