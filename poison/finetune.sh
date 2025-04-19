#!/bin/bash

# --- Interactive Script to Run poison_cnn.py with Finetuning ---

echo "Configure parameters for poison_cnn.py (Press Enter to use default values)"
echo "----------------------------------------------------------------------"

# --- Set Default Values ---
DEFAULT_POISON_RATIO=0.1
DEFAULT_SUBSET_RATIO=0.1
DEFAULT_EPOCHS=10
DEFAULT_LR=0.001
DEFAULT_BATCH_SIZE=128
DEFAULT_SEED=42
DEFAULT_NUM_WORKERS=2
DEFAULT_PLOT_EXAMPLES="n" # Default to No
DEFAULT_NO_CUDA="n"       # Default to No (use CUDA if available)

# --- Prompt User for Input ---

# Poison Ratio
read -p "Enter poison ratio [${DEFAULT_POISON_RATIO}]: " poison_ratio
poison_ratio=${poison_ratio:-$DEFAULT_POISON_RATIO}

# Subset Ratio
read -p "Enter dataset subset ratio (0.0 to 1.0) [${DEFAULT_SUBSET_RATIO}]: " subset_ratio
subset_ratio=${subset_ratio:-$DEFAULT_SUBSET_RATIO}

# Epochs
read -p "Enter number of epochs [${DEFAULT_EPOCHS}]: " epochs
epochs=${epochs:-$DEFAULT_EPOCHS}

# Learning Rate
read -p "Enter learning rate [${DEFAULT_LR}]: " lr
lr=${lr:-$DEFAULT_LR}

# Batch Size
read -p "Enter batch size [${DEFAULT_BATCH_SIZE}]: " batch_size
batch_size=${batch_size:-$DEFAULT_BATCH_SIZE}

# Random Seed
read -p "Enter random seed [${DEFAULT_SEED}]: " seed
seed=${seed:-$DEFAULT_SEED}

# Number of Workers
read -p "Enter number of data loader workers [${DEFAULT_NUM_WORKERS}]: " num_workers
num_workers=${num_workers:-$DEFAULT_NUM_WORKERS}

# Plot Examples Flag
read -p "Plot classification examples at the end? [y/N]: " plot_flag
plot_flag=${plot_flag:-$DEFAULT_PLOT_EXAMPLES}
plot_arg=""
if [[ "$plot_flag" =~ ^[Yy]$ ]]; then
  plot_arg="--plot_examples"
fi

# No CUDA Flag
read -p "Disable CUDA (force using CPU)? [y/N]: " no_cuda_flag
no_cuda_flag=${no_cuda_flag:-$DEFAULT_NO_CUDA}
cuda_arg=""
if [[ "$no_cuda_flag" =~ ^[Yy]$ ]]; then
  cuda_arg="--no_cuda"
fi

# --- Construct and Display the Command ---
cmd="python poison_cnn.py \
    --poison_ratio $poison_ratio \
    --subset_ratio $subset_ratio \
    --epochs $epochs \
    --lr $lr \
    --batch_size $batch_size \
    --seed $seed \
    --num_workers $num_workers \
    $plot_arg \
    $cuda_arg" # Add flags only if set

echo "----------------------------------------"
echo "Final Configuration:"
echo "  Poison Ratio: $poison_ratio"
echo "  Subset Ratio: $subset_ratio"
echo "  Epochs:       $epochs"
echo "  Learning Rate:$lr"
echo "  Batch Size:   $batch_size"
echo "  Seed:         $seed"
echo "  Num Workers:  $num_workers"
echo "  Plot Examples:$plot_flag"
echo "  Disable CUDA: $no_cuda_flag"
echo "----------------------------------------"
echo "Executing Command:"
# Use printf for better formatting control and to avoid potential issues with echo interpretation
printf "%s\n" "$cmd"
echo "----------------------------------------"

# --- Execute the Command ---
# Execute directly without eval for better security
python poison_cnn.py \
    --poison_ratio "$poison_ratio" \
    --subset_ratio "$subset_ratio" \
    --epochs "$epochs" \
    --lr "$lr" \
    --batch_size "$batch_size" \
    --seed "$seed" \
    --num_workers "$num_workers" \
    $plot_arg \
    $cuda_arg

echo "----------------------------------------"
echo "Script finished."