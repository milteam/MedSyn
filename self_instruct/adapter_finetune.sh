OUTPUT_DIR='lora-alpaca3'

docker run --gpus '"device=4,5"' --shm-size 64g -p 7860:7860 --name alpaca \
-v "${HOME}"/experiments/"${OUTPUT_DIR}":/workspace/"${OUTPUT_DIR}" \
--rm alpaca-lora \
  torchrun --nproc_per_node 2 adapter_finetune.py \
    --model Llama7B_adapter \
    --llama_model_path ./exps \
    --data_path 'alpaca_data_cleaned_archive.json' \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 5 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./$OUTPUT_DIR \
    --log_dir=./$OUTPUT_DIR
