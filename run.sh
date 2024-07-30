#!/bin/bash

# Base command
BASE_CMD="torchrun --standalone --nproc_per_node=8 run.py"

# Fixed parameters
FIXED_PARAMS="
    --input_bin \"data/fineweb10B/fineweb_train_*.bin\" \
    --input_val_bin \"data/fineweb10B/fineweb_val_*.bin\" \
    --batch_size 64 \
    --sequence_length 1024 \
    --val_loss_every 128 \
    --num_iterations 9536 \
    --weight_decay 0.1 \
    --warmup_iters 256 \
    --warmdown_iters 2048"

# Arrays for sweep parameters
WIDTHS=(1024 128)
loglr=(-10 -8 -6 -4 -2 0)

# Perform the sweep
for width in "${WIDTHS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        # Calculate number of heads (assuming head dimension of 64)
        n_head=$((width / 64))
        
        # Set up unique output directory and wandb run name
        OUTPUT_DIR="pylog_${width}_${lr}"
        WANDB_RUN_NAME="width_${width}_lr_${lr}"
        
        # Construct and execute the command
        CMD="$BASE_CMD \
            $FIXED_PARAMS \
            --output_dir $OUTPUT_DIR \
            --learning_rate $lr \
            --n_embd $width \
            --n_head $n_head \
            --wandb_project fineweb_sweep \
            --wandb_run_name $WANDB_RUN_NAME"
        
        echo "Running: $CMD"
        eval $CMD
        
        # Optional: add a delay between runs if needed
        # sleep 60
    done
done