#!/bin/bash

# Quick run script for EndoDora video generation
# Usage: bash run_generation.sh [input_image] [checkpoint] [output_dir]

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters
INPUT_IMAGE="${1:-$SCRIPT_DIR/seed_images/colonoscopy.jpg}"
CHECKPOINT="${2:-$PROJECT_ROOT/checkpoints/Colonoscopic_0150000.pt}"
OUTPUT_DIR="${3:-$SCRIPT_DIR/results}"

# Activate Endora environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Endora

echo "🎬 Running EndoDora Video Generation"
echo "=================================="
echo "Input image: $INPUT_IMAGE"
echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if files exist
if [[ ! -f "$INPUT_IMAGE" ]]; then
    echo "❌ Input image not found: $INPUT_IMAGE"
    echo "Available images in seed_images/:"
    ls -la "$SCRIPT_DIR/seed_images/" 2>/dev/null || echo "No images found"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -la "$PROJECT_ROOT/checkpoints/"*.pt 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# Run the generation
cd "$PROJECT_ROOT"
python SARA/generate_polyp_video.py \
    --config SARA/col_sample_cpu.yaml \
    --ckpt "$CHECKPOINT" \
    --input_image "$INPUT_IMAGE" \
    --save_video_path "$OUTPUT_DIR"

echo ""
echo "✅ Generation complete! Check the output in: $OUTPUT_DIR"
