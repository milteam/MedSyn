OUTPUT_DIR='models/ru_llama_7b_lora'

docker run --gpus '"device=0,1"' --name alpaca \
  -v "${HOME}"/.cache:/root/.cache \
  -v "${PWD}"/models/:/workspace/models/ \
  -v "${PWD}"/models/"$OUTPUT_DIR":/workspace/models/"$OUTPUT_DIR" \
  -v "${PWD}"/data/:/workspace/data \
  -v "${PWD}"/:/workspace/ \
  --rm alpaca-lora \
    python3.10 finetune.py \
      --base_model="models/meta-llama-v2-7b" \
      --resume_from_checkpoint="models/ru_saiga-v2_7b" \
      --only_target_loss=True \
      --data_path='data/all_data_merged_ift.jsonl' \
      --template_path="templates/ru_alpaca.json" \
      --model_type="causal" \
      --num_epochs=5 \
      --max_tokens_count=2048 \
      --learning_rate=2e-5 \
      --group_by_length \
      --output_dir=models/"$OUTPUT_DIR" \
      --lora_target_modules='[q_proj,v_proj,k_proj,o_proj]' \
      --lora_r=16 \
      --lora_alpha=16 \
      --batch_size=256\
      --micro_batch_size=4 \
      --warmup_steps=100 \
      --val_set_size=0
