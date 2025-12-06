#!/bin/bash

# =============================================================================
# ENDORA TEXT-GUIDED VIDEO GENERATION EXAMPLES
# =============================================================================
# 
# This script demonstrates different ways to use EndoDora for video generation:
# 1. Basic image-conditioned generation (no text prompts)
# 2. Text-guided generation with command-line prompts  
# 3. Text-guided generation using YAML configuration
# 4. High-quality generation with custom parameters
#
# Prerequisites:
# - Conda environment activated (Endora or endo5090)
# - Model checkpoint available
# - Input images in seed_images/ directory
#
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
if [[ ! -f "config.yaml" ]]; then
    print_error "config.yaml not found. Please run from SARA directory."
    exit 1
fi

if [[ ! -f "../checkpoints/Colonoscopic_0150000.pt" ]]; then
    print_error "Model checkpoint not found at ../checkpoints/Colonoscopic_0150000.pt"
    exit 1
fi

print_header "ENDORA VIDEO GENERATION EXAMPLES"
echo "This script demonstrates different generation modes with EndoDora."
echo ""

# =============================================================================
# EXAMPLE 1: Basic Image-Conditioned Generation (No Text Prompts)
# =============================================================================

print_header "EXAMPLE 1: Basic Image-Conditioned Generation"
print_info "Generating video from colonoscopy image without text prompts..."
print_info "Uses: Image conditioning only, no text guidance"

# Temporarily disable text guidance in config
sed -i 's/use_text_guidance: True/use_text_guidance: False/' config.yaml

python generate_polyp_video.py \
    --config config.yaml \
    --ckpt ../checkpoints/Colonoscopic_0150000.pt \
    --input_image seed_images/colonoscopy.jpg \
    --save_video_path ./results/

# Restore text guidance
sed -i 's/use_text_guidance: False/use_text_guidance: True/' config.yaml

print_info "✅ Example 1 complete: results/colonoscopy_generated_video.mp4"
echo ""

# =============================================================================
# EXAMPLE 2: Text-Guided Generation with Command-Line Prompts
# =============================================================================

print_header "EXAMPLE 2: Command-Line Text-Guided Generation"
print_info "Generating video with custom text prompts via command line..."
print_info "Uses: Image + text conditioning with CLI prompts"

python generate_polyp_video.py \
    --config config.yaml \
    --ckpt ../checkpoints/Colonoscopic_0150000.pt \
    --input_image seed_images/polyp.jpg \
    --save_video_path ./results/ \
    --prompt "smooth endoscope withdrawal, detailed polyp inspection, steady clinical lighting" \
    --negative_prompt "blurry camera movement, poor lighting, artifacts, low quality" \
    --use_text_guidance

print_info "✅ Example 2 complete: results/polyp_generated_video.mp4"
echo ""

# =============================================================================
# EXAMPLE 3: YAML Configuration-Based Text Guidance
# =============================================================================

print_header "EXAMPLE 3: YAML Configuration Text Guidance"
print_info "Generating video using prompts defined in config.yaml..."
print_info "Uses: Image + text conditioning from YAML config"

# Update config with clinical prompts
cat > temp_clinical_config.yaml << 'EOF'
# Clinical examination prompts
prompt: "gentle scope navigation, clear mucosal visualization, optimal illumination, professional endoscopy technique"
negative_prompt: "shaky movement, dark shadows, motion blur, poor image quality, artifacts"
use_text_guidance: True

# High quality settings
num_frames: 20
num_sampling_steps: 300
cfg_scale: 8.0
fps: 10
video_quality: 9
EOF

# Merge with existing config (keeping other parameters)
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open('temp_clinical_config.yaml', 'r') as f:
    clinical = yaml.safe_load(f)
config.update(clinical)
with open('config_clinical.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"

python generate_polyp_video.py \
    --config config_clinical.yaml \
    --ckpt ../checkpoints/Colonoscopic_0150000.pt \
    --input_image seed_images/multiple_polyps.jpg \
    --save_video_path ./results/

# Cleanup temporary files
rm temp_clinical_config.yaml config_clinical.yaml

print_info "✅ Example 3 complete: results/multiple_polyps_generated_video.mp4"
echo ""

# =============================================================================
# EXAMPLE 4: High-Quality Generation with Custom Parameters
# =============================================================================

print_header "EXAMPLE 4: High-Quality Custom Generation"
print_info "Generating high-quality video with custom parameters..."
print_info "Uses: Extended frames, higher resolution, more sampling steps"

# Create high-quality config
cat > hq_config.yaml << EOF
# Inherit from base config
$(cat config.yaml)

# High-quality overrides
num_frames: 24              # Longer video
image_size: 256             # Higher resolution  
num_sampling_steps: 500     # More denoising steps
cfg_scale: 10.0             # Stronger guidance
fps: 12                     # Smoother playback
video_quality: 10           # Best compression quality

# Advanced clinical prompts
prompt: "smooth endoscopic examination, detailed anatomical visualization, professional camera control, optimal clinical lighting, high definition imagery"
negative_prompt: "camera shake, motion blur, poor illumination, low resolution, compression artifacts, unprofessional technique"
use_text_guidance: True

# Performance settings for high-quality
per_proc_batch_size: 2      # Reduced batch size for higher resolution
use_fp16: True              # Enable mixed precision
EOF

print_warning "High-quality mode requires more GPU memory and processing time..."
print_info "Expected generation time: 10-15 seconds on RTX 5090"

python generate_polyp_video.py \
    --config hq_config.yaml \
    --ckpt ../checkpoints/Colonoscopic_0150000.pt \
    --input_image seed_images/polip1.jpg \
    --save_video_path ./results/

rm hq_config.yaml

print_info "✅ Example 4 complete: results/polip1_generated_video.mp4"
echo ""

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

print_header "GENERATION RESULTS SUMMARY"

echo "Generated Videos:"
echo "=================="
ls -lh results/*.mp4 | tail -4

echo ""
echo "File Sizes and Details:"
echo "======================="
for video in results/*.mp4; do
    if [[ -f "$video" ]]; then
        size=$(du -h "$video" | cut -f1)
        echo "• $(basename "$video"): $size"
    fi
done

echo ""
print_info "🎬 Convert to GIFs for README display:"
echo "   cd results && bash convert_to_gif.sh"
echo ""
print_info "📊 Video Analysis:"
echo "   • Basic generation: Fastest, image-only conditioning"
echo "   • CLI text guidance: Custom prompts, real-time control"  
echo "   • YAML text guidance: Reproducible, configuration-driven"
echo "   • High-quality mode: Best quality, longer processing time"
echo ""
print_info "🔧 Configuration Tips:"
echo "   • Increase num_sampling_steps (250→500) for better quality"
echo "   • Increase cfg_scale (7.5→15.0) for stronger conditioning" 
echo "   • Increase num_frames (16→32) for longer videos"
echo "   • Increase image_size (128→256) for higher resolution"
echo ""
print_info "✅ All examples completed successfully!"