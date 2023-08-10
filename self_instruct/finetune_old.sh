#BASE_MODEL='models/ru_decapoda-llama-7b'
#BASE_MODEL='models/ru_huggyllama-llama-7b'
BASE_MODEL='models/rugpt-13b'
#OUTPUT_DIR='models/ru_turbo-alpaca-med-data_512_qv_1e-5'
OUTPUT_DIR='models/rugpt-13b-gigasaiga-med-data'


    #--resume_from_checkpoint=$BASE_MODEL \
    #--lora_target_modules='[q_proj,v_proj,k_proj,o_proj]' \
    
docker run --gpus '"device=5,6"' --shm-size 64g -p 7860:7860 --name alpaca \
  -v "${HOME}"/.cache:/root/.cache \
  -v "${HOME}"/MedTexts/alpaca-lora/"${OUTPUT_DIR}":/workspace/"${OUTPUT_DIR}" --rm alpaca-lora \
  python3.10 finetune.py \
    --base_model=$BASE_MODEL \
    --resume_from_checkpoint=$BASE_MODEL \
    --prompt_template_name='ru_alpaca' \
    --num_epochs=3 \
    --val_set_size=0 \
    --warmup_steps=10 \
    --cutoff_len=512 \
    --learning_rate=1e-5 \
    --group_by_length \
    --data_path='alpaca_med_data.json' \
    --output_dir=./$OUTPUT_DIR \
    --lora_target_modules='[c_attn]' \
    --lora_r=16 \
    --lora_alpha=16 \
    --micro_batch_size=4
