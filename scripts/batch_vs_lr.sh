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

# Arrays for sweeping
LEARNING_RATES=(0.01 0.0031 0.001 0.0031)
GRAD_ACCUM_SIZES=(2 4 8 16)

# Loop over learning rates and gradient accumulation sizes
for lr in "${LEARNING_RATES[@]}"; do
    for grad_accum in "${GRAD_ACCUM_SIZES[@]}"; do
        echo "Running with learning rate: $lr and gradient accumulation: $grad_accum"
        
        # Calculate effective batch size
        effective_batch_size=$((BATCH_SIZE * grad_accum))
        
        # Create a unique output directory for each run
        OUTPUT_DIR="log_100m_lr${lr}_gradaccum${grad_accum}"

        echo "Output directory: $OUTPUT_DIR, effective batch size: $effective_batch_size, adjusted iterations: $adjusted_iterations"
        
        # Run the training script
        torchrun --standalone --nproc_per_node=8 run.py \
            --input_bin "$INPUT_BIN" \
            --input_val_bin "$INPUT_VAL_BIN" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size $BATCH_SIZE \
            --sequence_length $SEQUENCE_LENGTH \
            --val_loss_every $VAL_LOSS_EVERY \
            --num_iterations $NUM_ITERATIONS \
            --weight_decay $WEIGHT_DECAY \
            --n_embd 128 \
            --n_head 8 \
            --learning_rate $lr \
            --warmup_iters $WARMUP_ITERS \
            --warmdown_iters $WARMDOWN_ITERS \
            --gradient_accumulation_steps $grad_accum \
            --wandb_run_name "lr${lr}_gradaccum${grad_accum}"
    done
done