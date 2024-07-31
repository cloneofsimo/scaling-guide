torchrun --standalone --nproc_per_node=8 run.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir log_100m \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --sequence_length 1024 \
    --val_loss_every 128 \
    --num_iterations 9536 \
    --weight_decay 0.1 \
    --learning_rate 0.001 \
    --warmup_iters 256 \
    --warmdown_iters 2048