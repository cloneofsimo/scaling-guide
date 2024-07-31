#!/bin/bash

# Fixed parameters
INPUT_BIN="data/fineweb10B/fineweb_train_*.bin"
INPUT_VAL_BIN="data/fineweb10B/fineweb_val_*.bin"
BATCH_SIZE=32
SEQUENCE_LENGTH=1024
VAL_LOSS_EVERY=128
NUM_ITERATIONS=9536
WEIGHT_DECAY=0.1
WARMUP_ITERS=256
WARMDOWN_ITERS=2048
N_LAYER=8

LEARNING_RATES=(0.01 0.0031 0.001 0.0031)
MODEL_WIDTHS=(64 256 1024)

# Loop over learning rates and model widths
for lr in "${LEARNING_RATES[@]}"; do
    for width in "${MODEL_WIDTHS[@]}"; do
        echo "Running with learning rate: $lr and model width: $width"
        
        # Calculate number of heads and layers based on width
        n_head=$((width / 64))

        # Create a unique output directory for each run
        OUTPUT_DIR="log_100m_lr${lr}_width${width}"
        
        # Run the training script
        torchrun --standalone --nproc_per_node=8 run.py \
            --wandb_project fineweb_lr_width_sweep \
            --input_bin "$INPUT_BIN" \
            --input_val_bin "$INPUT_VAL_BIN" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size $BATCH_SIZE \
            --sequence_length $SEQUENCE_LENGTH \
            --val_loss_every $VAL_LOSS_EVERY \
            --num_iterations $NUM_ITERATIONS \
            --weight_decay $WEIGHT_DECAY \
            --learning_rate $lr \
            --warmup_iters $WARMUP_ITERS \
            --warmdown_iters $WARMDOWN_ITERS \
            --n_embd $width \
            --n_head $n_head \
            --n_layer $N_LAYER \
            --wandb_run_name "lr${lr}_width${width}"
    done
done