#!/bin/bash

# =============================================================================
# CREATE 512x512 VISUALIZATIONS FOR README
# =============================================================================
# 
# This script creates properly sized visualizations for the README:
# 1. Resizes seed images to 512x512 for consistent display
# 2. Converts generated videos to 512x512 GIFs
# 3. Ensures uniform presentation in documentation
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

print_header "CREATING 512x512 VISUALIZATIONS FOR README"

# Check prerequisites
if ! command -v ffmpeg &> /dev/null; then
    print_error "ffmpeg is not installed. Please install it (e.g., sudo apt install ffmpeg)."
    exit 1
fi

if ! command -v convert &> /dev/null; then
    print_error "ImageMagick convert is not installed. Please install it (e.g., sudo apt install imagemagick)."
    exit 1
fi

# Create directories for visualizations
mkdir -p visualizations/seed_images_512x512
mkdir -p visualizations/generated_gifs_512x512

print_info "📁 Created visualization directories"

# =============================================================================
# RESIZE SEED IMAGES TO 512x512
# =============================================================================

print_header "RESIZING SEED IMAGES TO 512x512"

SEED_IMAGES=(
    "colonoscopy.jpg"
    "polyp.jpg" 
    "polip1.jpg"
    "multiple_polyps.jpg"
)

for img in "${SEED_IMAGES[@]}"; do
    if [[ -f "seed_images/$img" ]]; then
        print_info "🖼️  Resizing seed_images/$img to 512x512..."
        convert "seed_images/$img" -resize 512x512 -gravity center -background white -extent 512x512 "visualizations/seed_images_512x512/$img"
        
        # Get file size
        SIZE=$(du -h "visualizations/seed_images_512x512/$img" | cut -f1)
        print_info "✅ Created: visualizations/seed_images_512x512/$img ($SIZE)"
    else
        print_warning "⚠️  Seed image not found: seed_images/$img"
    fi
done

# =============================================================================
# CONVERT VIDEOS TO 512x512 GIFS
# =============================================================================

print_header "CONVERTING VIDEOS TO 512x512 GIFS"

VIDEO_FILES=$(find results/ -name "*.mp4" -type f)
if [[ -z "$VIDEO_FILES" ]]; then
    print_error "No MP4 files found in results/ directory"
    exit 1
fi

for video in $VIDEO_FILES; do
    basename_video=$(basename "$video" .mp4)
    gif_output="visualizations/generated_gifs_512x512/${basename_video}.gif"
    
    print_info "🎬 Converting: $video -> $gif_output"
    print_info "   Settings: 512x512 resolution, 8fps, optimized for README display"
    
    # Create high-quality 512x512 GIF
    ffmpeg -y -i "$video" \
        -vf "fps=8,scale=512:512:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
        -loop 0 \
        "$gif_output" > /dev/null 2>&1
    
    if [[ $? -eq 0 ]]; then
        GIF_SIZE=$(du -h "$gif_output" | cut -f1)
        VIDEO_SIZE=$(du -h "$video" | cut -f1)
        print_info "✅ Created: $gif_output ($GIF_SIZE, original: $VIDEO_SIZE)"
    else
        print_error "❌ Failed to convert $video"
    fi
done

# =============================================================================
# CREATE VISUALIZATION SUMMARY
# =============================================================================

print_header "VISUALIZATION SUMMARY"

echo "512x512 Seed Images:"
echo "===================="
ls -lh visualizations/seed_images_512x512/

echo ""
echo "512x512 Generated GIFs:"
echo "========================"
ls -lh visualizations/generated_gifs_512x512/

echo ""
print_info "📊 File Count Summary:"
SEED_COUNT=$(ls visualizations/seed_images_512x512/ | wc -l)
GIF_COUNT=$(ls visualizations/generated_gifs_512x512/ | wc -l)
print_info "   • Seed images (512x512): $SEED_COUNT"
print_info "   • Generated GIFs (512x512): $GIF_COUNT"

echo ""
print_info "💡 Usage in README.md:"
echo "   ![Seed Image](SARA/visualizations/seed_images_512x512/colonoscopy.jpg)"
echo "   ![Generated GIF](SARA/visualizations/generated_gifs_512x512/colonoscopy_generated_video.gif)"

echo ""
print_info "🎯 These visualizations provide:"
print_info "   • Consistent 512x512 size for side-by-side comparison"
print_info "   • Optimized file sizes for web display"
print_info "   • Professional presentation in documentation"
print_info "   • Clear visual contrast between seed and generated content"

print_info "✅ 512x512 visualizations created successfully!"
