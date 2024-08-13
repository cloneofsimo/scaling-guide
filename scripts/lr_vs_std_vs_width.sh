#!/bin/bash

# Fixed parameters
INPUT_BIN="data/fineweb10B/fineweb_train_*.bin"
INPUT_VAL_BIN="data/fineweb10B/fineweb_val_*.bin"
BATCH_SIZE=64
SEQUENCE_LENGTH=512
VAL_LOSS_EVERY=1024
NUM_ITERATIONS=10240
WEIGHT_DECAY=0.1
WARMUP_ITERS=128
WARMDOWN_ITERS=1024
N_LAYER=12

LEARNING_RATES=(0.316 0.1 0.0316 0.01 0.00316 0.001 0.000316 0.0001)
#LEARNING_RATES=(0.000316 0.0001)
MODEL_WIDTHS=(64 256)
STD=(0.0316 0.1 0.316 1.0 3.16 10.0)

# Loop over learning rates and model widths
for width in "${MODEL_WIDTHS[@]}"; do
    for std in "${STD[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            echo "Running with learning rate: $lr and model width: $width and std: $std"
            
            # Calculate number of heads and layers based on width
            n_head=$((width / 8))

            # Create a unique output directory for each run
            OUTPUT_DIR="./ckpts/log_100m_lr${lr}_width${width}_std${std}"
            
            # Run the training script
            torchrun --standalone --nproc_per_node=8 run.py \
                --wandb_project exp2_mup_lr_width_std \
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
                --wandb_run_name "lr${lr}_width${width}_std{std}_2" \
                --val_max_steps 40 \
                --gpt_linear_init_std $std \
                --wandb_tags "std-lr-width-sweep"
        done
    done
done