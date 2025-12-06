#!/bin/bash
# =============================================================================
# DynamiCrafter LoRA Fine-tuning Training Script
# Runs LoRA fine-tuning on DynamiCrafter for watermark elimination
# =============================================================================

set -e  # Exit on any error

echo "🚀 DynamiCrafter LoRA Fine-tuning Training"
echo "=========================================="

# Configuration
BASE_MODEL="/ops/model_checkpoints/diffusion_models/dynamicrafter_model.ckpt"
CONFIG_FILE="config_lora.yaml"
DATA_ROOT="/ops/datasets/colonoscopy"
OUTPUT_DIR="./lora_checkpoints"
EXPERIMENT_NAME="dynamicrafter_colonoscopy_lora"

# Hardware settings
GPUS=1
BATCH_SIZE=4
NUM_WORKERS=4
PRECISION="16-mixed"

# LoRA settings
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Training settings
LEARNING_RATE=1e-4
MAX_EPOCHS=100
MAX_STEPS=50000

# Validate inputs
echo "🔍 Validating configuration..."

if [ ! -f "$BASE_MODEL" ]; then
    echo "❌ Error: Base model not found at $BASE_MODEL"
    echo "💡 Please ensure the DynamiCrafter checkpoint is available"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️  Warning: Dataset directory not found at $DATA_ROOT"
    echo "🛠️  Creating dummy dataset structure..."
    python -c "
import sys
sys.path.append('.')
from dataset import create_dummy_dataset
create_dummy_dataset('$DATA_ROOT')
"
    echo "📁 Dummy dataset created. Please add your colonoscopy videos."
    echo "💡 You can continue training with the dummy data for testing."
fi

# Check GPU availability
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🔍 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  Warning: nvidia-smi not found. Running on CPU."
    GPUS=0
    PRECISION="32"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables for optimization
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "🔧 Activating conda environment: $CONDA_ENV"
    source activate "$CONDA_ENV"
fi

echo "📋 Training Configuration:"
echo "   Base Model: $BASE_MODEL"
echo "   Config: $CONFIG_FILE"  
echo "   Dataset: $DATA_ROOT"
echo "   Output: $OUTPUT_DIR"
echo "   Experiment: $EXPERIMENT_NAME"
echo ""
echo "🎯 LoRA Settings:"
echo "   Rank: $LORA_RANK"
echo "   Alpha: $LORA_ALPHA"
echo "   Dropout: $LORA_DROPOUT"
echo ""
echo "⚙️  Hardware Settings:"
echo "   GPUs: $GPUS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Precision: $PRECISION"
echo "   Workers: $NUM_WORKERS"
echo ""

# Start training
echo "🚀 Starting LoRA fine-tuning..."
echo "⏰ Started at: $(date)"

python train_lora.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$BASE_MODEL" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --video_length 16 \
    --resolution 256 \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --learning_rate $LEARNING_RATE \
    --max_epochs $MAX_EPOCHS \
    --max_steps $MAX_STEPS \
    --gpus $GPUS \
    --precision "$PRECISION"

echo ""
echo "✅ Training completed at: $(date)"
echo "📁 Checkpoints saved to: $OUTPUT_DIR"
echo "📊 TensorBoard logs: $OUTPUT_DIR/lightning_logs"
echo ""
echo "🎯 Next steps:"
echo "   1. Check TensorBoard: tensorboard --logdir $OUTPUT_DIR/lightning_logs"
echo "   2. Test inference: python inference_lora.py --checkpoint $OUTPUT_DIR/last.ckpt"
echo "   3. Evaluate results: Compare with original model outputs"
echo ""
echo "🎉 Happy fine-tuning!"
