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
