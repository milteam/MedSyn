OUTPUT_DIR='ru_llama_7b_lora'

docker run --gpus '"device=4,5"' --shm-size 64g -p 7860:7860 --name alpaca \
  -v "${HOME}"/.cache:/root/.cache \
  -v "${HOME}"/MedTexts/alpaca-lora/models/"$OUTPUT_DIR":/workspace/models/"$OUTPUT_DIR" \
  --rm alpaca-lora \
  python3.10 finetune.py \
    --base_model='models/decapoda-llama-7b' \
    --resume_from_checkpoint="models/ru_turbo_alpaca_7b" \
    --only_target_loss=False \
    --data_path='data/alpaca_med_data_10k.json' \
    --template_path="templates/ru_alpaca.json" \
    --model_type="causal" \
    --mode="instruct" \
    --num_epochs=3 \
    --max_source_tokens_count=256 \
    --max_target_tokens_count=512 \
    --learning_rate=1e-5 \
    --group_by_length \
    --output_dir=models/"$OUTPUT_DIR" \
    --lora_target_modules='[q_proj,v_proj]' \
    --lora_r=8 \
    --lora_alpha=16 \
    --micro_batch_size=8 \
    --warmup_steps=10 \
    --val_set_size=0
