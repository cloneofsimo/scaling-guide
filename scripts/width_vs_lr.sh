#!/bin/bash

# Fixed parameters
INPUT_BIN="data/fineweb10B/fineweb_train_*.bin"
INPUT_VAL_BIN="data/fineweb10B/fineweb_val_*.bin"
BATCH_SIZE=128
SEQUENCE_LENGTH=512
VAL_LOSS_EVERY=512
NUM_ITERATIONS=10000
WEIGHT_DECAY=0.0
WARMUP_ITERS=128
WARMDOWN_ITERS=128
N_LAYER=8

LEARNING_RATES=(0.0316 0.01 0.00316 0.001)
MODEL_WIDTHS=(64 256 512)

# Loop over learning rates and model widths
for width in "${MODEL_WIDTHS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo "Running with learning rate: $lr and model width: $width"
        
        # Calculate number of heads and layers based on width
        n_head=$((width / 16))

        # Create a unique output directory for each run
        OUTPUT_DIR="./ckpts/log_100m_lr${lr}_width${width}"
        
        # Run the training script
        torchrun --standalone --nproc_per_node=8 run.py \
            --wandb_project a1008xnsml_lr_width_sweep_muP_fixed \
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
            --wandb_run_name "lr${lr}_width${width}_2" \
            --val_max_steps 120 \
            --wandb_tags "wd0.0" \

    done
done