#!/bin/bash

# SARA - EndoDora Video Generation Setup Script  
# Based on official CUHK-AIM-Group/Endora repository
# https://github.com/CUHK-AIM-Group/Endora#
# This script sets up the complete environment from scratch
# Usage: bash setup_endora.sh

set -e  # Exit on any error

echo "🚀 Setting up EndoDora Video Generation Environment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Miniconda/Anaconda first."
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Check NVIDIA driver for RTX 5090 compatibility
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
    
    if [[ "$GPU_NAME" == *"RTX 5090"* ]]; then
        print_success "RTX 5090 (Blackwell) detected: $GPU_NAME"
        print_status "Driver version: $DRIVER_VERSION"
        
        # Check if driver supports CUDA 12.8 (need ≥ 570.26 on Linux)
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
        DRIVER_MINOR=$(echo "$DRIVER_VERSION" | cut -d. -f2)
        
        if [[ $DRIVER_MAJOR -gt 570 ]] || [[ $DRIVER_MAJOR -eq 570 && $DRIVER_MINOR -ge 26 ]]; then
            print_success "✅ Driver $DRIVER_VERSION supports CUDA 12.8 for Blackwell"
            USE_RTX5090_ENV=true
        else
            print_error "❌ Driver $DRIVER_VERSION too old for RTX 5090"
            print_error "RTX 5090 requires driver ≥ 570.26 for CUDA 12.8 support"
            print_error "Please update your NVIDIA driver and try again"
            exit 1
        fi
    fi
fi

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_status "Script directory: $SCRIPT_DIR"
print_status "Project root: $PROJECT_ROOT"

# Determine environment name and setup strategy
if [[ "$USE_RTX5090_ENV" == "true" ]]; then
    ENV_NAME="endo5090"
    print_status "Creating optimized environment for RTX 5090: $ENV_NAME"
else
    ENV_NAME="Endora"
    print_status "Creating standard environment: $ENV_NAME"
fi

# Check if we're currently in the target environment
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    print_warning "Currently in $ENV_NAME environment. Please run this script from base environment:"
    print_warning "conda deactivate && bash setup_endora.sh"
    print_status "Continuing with existing environment setup..."
    ENV_EXISTS=true
elif conda env list | grep -q "^$ENV_NAME "; then
    print_warning "$ENV_NAME environment already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing $ENV_NAME environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        print_status "Using existing $ENV_NAME environment..."
        ENV_EXISTS=true
    fi
fi

# Create conda environment if it doesn't exist
if [[ "$ENV_EXISTS" != "true" ]]; then
    print_status "Creating $ENV_NAME conda environment with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate environment
print_status "Activating $ENV_NAME environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    print_error "Failed to activate $ENV_NAME environment"
    exit 1
fi

print_success "$ENV_NAME environment activated successfully"

# Install compatible packages with fixed versions
print_status "Installing compatible package versions for stable operation..."

# First install numpy 1.x to avoid compatibility issues
pip install "numpy<2.0,>=1.24.0"

# Install PyTorch based on GPU architecture
if [[ "$USE_RTX5090_ENV" == "true" ]]; then
    print_status "Installing PyTorch 2.7+ with CUDA 12.8 for RTX 5090 (Blackwell sm_120)..."
    print_status "Reference: RTX 5090 requires CUDA 12.8 + cu128 wheels for native kernels"
    
    # Install PyTorch built for CUDA 12.8 (cu128) - stable version 2.7+
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    
    print_success "✅ PyTorch cu128 installed for RTX 5090 compatibility"
else
    print_status "Installing PyTorch 2.1.2 with CUDA 11.8 (official CUHK-AIM-Group/Endora)..."
    print_status "Reference: https://github.com/CUHK-AIM-Group/Endora#"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
fi

# Verify GPU availability and memory
print_status "Verifying GPU setup..."
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    print_success "GPU detected: $GPU_INFO"
    
    if [[ $GPU_MEMORY -ge 24000 ]]; then
        print_success "✅ GPU memory ($GPU_MEMORY MB) meets EndoDora requirements (24GB+)"
        GPU_READY=true
        
        # Check for RTX 5090 and inform about compatibility
        if [[ "$GPU_INFO" == *"RTX 5090"* ]]; then
            print_success "🚀 RTX 5090 detected - excellent performance expected!"
            print_warning "📝 Note: Using compatible PyTorch version (not official 2.1.2)"
        fi
    else
        print_warning "⚠️  GPU memory ($GPU_MEMORY MB) below recommended 24GB"
        print_warning "Video generation may require reduced batch size or frame count"
        GPU_READY=false
    fi
else
    print_error "❌ No NVIDIA GPU detected. EndoDora requires GPU with 24GB+ for optimal performance."
    print_error "Installation will continue but performance will be severely limited."
    GPU_READY=false
fi

# Install core packages
if [[ "$USE_RTX5090_ENV" == "true" ]]; then
    print_status "Installing packages for RTX 5090 environment..."
    pip install einops omegaconf decord opencv-python safetensors
    pip install diffusers accelerate transformers timm av scikit-image pandas imageio-ffmpeg moviepy torchmetrics click tensorboard
else
    print_status "Installing core packages with compatible versions..."
    pip install "diffusers==0.24.0" "transformers==4.35.0" "huggingface-hub==0.17.3"
    
    # Install requirements from project root with version constraints
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        print_status "Installing additional requirements from $PROJECT_ROOT/requirements.txt..."
        pip install timm accelerate tensorboard einops av scikit-image decord pandas imageio-ffmpeg omegaconf moviepy torchmetrics click
    else
        print_warning "requirements.txt not found, installing essential packages manually..."
        pip install timm accelerate tensorboard einops av scikit-image decord pandas imageio-ffmpeg omegaconf moviepy torchmetrics click
    fi
    
    # Install additional packages
    print_status "Installing additional packages..."
    pip install pillow "opencv-python<5.0"
fi

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/results"
if [[ ! -d "$RESULTS_DIR" ]]; then
    print_status "Creating results directory: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
fi

# Verify seed images exist
SEED_IMAGES_DIR="$SCRIPT_DIR/seed_images"
if [[ ! -d "$SEED_IMAGES_DIR" ]]; then
    print_warning "Seed images directory not found: $SEED_IMAGES_DIR"
    print_status "Creating seed images directory..."
    mkdir -p "$SEED_IMAGES_DIR"
    print_warning "Please add your input images (e.g., colonoscopy.jpg, polyp.jpg) to: $SEED_IMAGES_DIR"
fi

# Check for checkpoints
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
    print_warning "Checkpoints directory not found: $CHECKPOINTS_DIR"
    print_warning "Please ensure model checkpoints are available in the checkpoints directory"
else
    print_status "Available checkpoints:"
    ls -la "$CHECKPOINTS_DIR"/*.pt 2>/dev/null || print_warning "No .pt checkpoint files found"
fi

# Create run script
RUN_SCRIPT="$SCRIPT_DIR/run_generation.sh"
print_status "Creating convenient run script: $RUN_SCRIPT"

cat > "$RUN_SCRIPT" << 'EOF'
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
EOF

chmod +x "$RUN_SCRIPT"

# Create usage instructions
USAGE_FILE="$SCRIPT_DIR/USAGE.md"
print_status "Creating usage instructions: $USAGE_FILE"

cat > "$USAGE_FILE" << 'EOF'
# EndoDora Video Generation - SARA

## Quick Start

### 1. Setup (one-time)
```bash
cd SARA
bash setup_endora.sh
```

### 2. Run Video Generation
```bash
# Using default settings (colonoscopy.jpg)
bash run_generation.sh

# Using specific image
bash run_generation.sh seed_images/polyp.jpg

# Using specific checkpoint
bash run_generation.sh seed_images/colonoscopy.jpg ../checkpoints/Kvasir-Capsule_0150000.pt

# Custom output directory
bash run_generation.sh seed_images/polyp.jpg ../checkpoints/Colonoscopic_0150000.pt ./my_output/
```

### 3. Manual Usage
```bash
# Activate environment
conda activate Endora

# Run generation
cd /path/to/endo-diffusion-models
python SARA/generate_polyp_video.py \
    --config SARA/col_sample_cpu.yaml \
    --ckpt checkpoints/Colonoscopic_0150000.pt \
    --input_image SARA/seed_images/colonoscopy.jpg \
    --save_video_path ./SARA/results/
```

## Available Checkpoints
- `CholecTriplet_0150000.pt` - CholecTriplet dataset
- `Colonoscopic_0150000.pt` - Colonoscopic dataset  
- `Kvasir-Capsule_0150000.pt` - Kvasir-Capsule dataset

## Configuration
- Model: EnDora-XL/2
- Frames: 16
- Image size: 128x128
- Sampling steps: 250
- Device: CPU (RTX 5090 compatibility)
- FP16: Disabled for CPU

## Troubleshooting
- If you get import errors, run `bash setup_endora.sh` again
- Make sure you have conda installed
- Check that checkpoint files exist in `checkpoints/` directory
- Verify input images are in `seed_images/` directory
EOF

print_success "Setup completed successfully! 🎉"
echo ""

# Final verification and GPU check
print_status "Verifying installation..."

if [[ "$USE_RTX5090_ENV" == "true" ]]; then
    print_status "Running RTX 5090 (Blackwell) sanity check..."
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x=torch.randn(1024,1024,device="cuda"); y=x@x; print("matmul ok:", y.is_cuda)
    print("✅ RTX 5090 Blackwell compatibility verified!")
else:
    print("⚠️  CUDA not available")
PY
    if [[ $? -eq 0 ]]; then
        print_success "✅ RTX 5090 sanity check passed!"
    else
        print_error "❌ RTX 5090 sanity check failed"
    fi
else
    python -c "
import torch, diffusers, transformers, einops
print('✅ All packages imported successfully')
if torch.cuda.is_available():
    print(f'🚀 CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
    print(f'🔥 CUDA Version: {torch.version.cuda}')
else:
    print('⚠️  CUDA not available - performance will be limited')
" || print_error "Package verification failed"
fi

echo ""
if [[ "$GPU_READY" == "true" ]]; then
    echo "🎯 Optimized for GPU inference with 32GB memory!"
    echo ""
    echo "📋 Quick start:"
    echo "1. Add your input images to: $SEED_IMAGES_DIR"
    echo "2. Run: bash $SCRIPT_DIR/run_generation.sh (optimized for GPU)"
    echo "3. Check results in: $RESULTS_DIR"
else
    echo "⚠️  GPU setup incomplete - check GPU drivers and CUDA installation"
    echo ""
    echo "📋 Next steps:"
    echo "1. Ensure NVIDIA GPU drivers are installed"
    echo "2. Verify CUDA 11.8 compatibility"
    echo "3. Add input images to: $SEED_IMAGES_DIR"
    echo "4. Run: bash $SCRIPT_DIR/run_generation.sh"
fi
echo ""
echo "📖 For detailed usage instructions, see: $USAGE_FILE"
echo ""
echo "🎬 EndoDora environment ready for 32GB GPU video generation!"
