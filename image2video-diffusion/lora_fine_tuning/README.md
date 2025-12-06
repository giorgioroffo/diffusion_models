# DynamiCrafter LoRA Fine-tuning

**Eliminate watermarks and adapt DynamiCrafter for medical colonoscopy videos using LoRA (Low-Rank Adaptation) fine-tuning.**

## 🎯 Overview

This directory contains a complete pipeline for fine-tuning DynamiCrafter using LoRA on clean colonoscopy data to eliminate watermarks like "Shutterstock" that appear in generated videos. LoRA allows efficient fine-tuning by training only ~1% of model parameters while preserving the base model's capabilities.

## 📁 Project Structure

```
lora_fine_tuning/
├── README.md                 # This file
├── requirements_lora.txt     # Additional Python dependencies
├── config_lora.yaml         # Training configuration
├── lora_wrapper.py          # LoRA implementation for DynamiCrafter
├── dataset.py               # Colonoscopy dataset loader
├── train_lora.py           # Main training script
├── inference_lora.py       # Inference with LoRA model
├── run_training.sh         # Training automation script
└── lora_checkpoints/       # Output directory (created during training)
```

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
# Install additional dependencies
pip install -r requirements_lora.txt

# Make training script executable
chmod +x run_training.sh
```

### 2. **Prepare Dataset**

Create your colonoscopy dataset in the expected structure:

```bash
/ops/datasets/colonoscopy/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── annotations/
│   ├── video_001.json
│   ├── video_002.json
│   └── ...
└── metadata.json
```

**Or create a dummy dataset for testing:**

```bash
python dataset.py
```

### 3. **Start Training**

```bash
# Automated training (recommended)
./run_training.sh

# Or manual training
python train_lora.py \
    --config config_lora.yaml \
    --checkpoint /ops/model_checkpoints/diffusion_models/dynamicrafter_model.ckpt \
    --data_root /ops/datasets/colonoscopy
```

### 4. **Generate Videos**

```bash
python inference_lora.py \
    --lora_checkpoint ./lora_checkpoints/last.ckpt \
    --image sample_imgs/large_polyp.jpg \
    --prompt "clean colonoscopy examination without watermarks" \
    --output clean_colonoscopy_video.mp4
```

## 🔧 Configuration Guide

### **Key Parameters in `config_lora.yaml`:**

#### **LoRA Settings:**
```yaml
lora:
  rank: 16              # LoRA rank (4, 8, 16, 32) - higher = more capacity
  alpha: 32.0           # Scaling factor - controls adaptation strength  
  dropout: 0.1          # Regularization
```

#### **Training Settings:**
```yaml
training:
  learning_rate: 1.0e-4 # Learning rate for LoRA parameters
  batch_size: 4         # Adjust based on GPU memory
  max_epochs: 100       # Training duration
```

#### **Hardware Settings:**
```yaml
hardware:
  gpus: 1               # Number of GPUs
  precision: "16-mixed" # Mixed precision for efficiency
```

## 📊 Monitoring Training

### **TensorBoard Logs:**
```bash
tensorboard --logdir ./lora_checkpoints/lightning_logs
```

### **Key Metrics to Watch:**
- **`train_loss`**: Should decrease steadily
- **`val_loss`**: Should follow train_loss without large gaps
- **`learning_rate`**: Should follow cosine annealing schedule
- **Generated samples**: Visual quality improvement over epochs

## 🎯 Expected Results

### **Before LoRA Fine-tuning:**
- ❌ Watermarks appear after ~30 frames
- ❌ Generic video generation
- ❌ "Shutterstock" artifacts

### **After LoRA Fine-tuning:**
- ✅ **No watermarks** in generated videos
- ✅ **Medical domain specialization**
- ✅ **Better colonoscopy-specific generation**
- ✅ **Preserved base model capabilities**

## 📈 Performance Optimization

### **For Better Quality:**
```yaml
# Increase LoRA capacity
lora:
  rank: 32
  alpha: 64.0

# More training steps
training:
  max_steps: 100000
  max_epochs: 200
```

### **For Faster Training:**
```yaml
# Reduce model size
data:
  batch_size: 8
  video_length: 12

# Faster validation
training:
  val_check_interval: 0.5
```

### **For Memory Efficiency:**
```yaml
# Enable CPU offloading
hardware:
  use_cpu_offload: true
  
# Gradient accumulation
training:
  accumulate_grad_batches: 2
```

## 🛠️ Troubleshooting

### **Common Issues:**

#### **1. GPU Out of Memory:**
```bash
# Reduce batch size
--batch_size 2

# Use gradient accumulation
--accumulate_grad_batches 2

# Enable CPU offload
--use_cpu_offload
```

#### **2. Dataset Not Found:**
```bash
# Create dummy dataset
python dataset.py

# Check dataset structure
ls -la /ops/datasets/colonoscopy/
```

#### **3. LoRA Not Applied:**
```bash
# Check target modules in config
target_modules: ["to_q", "to_k", "to_v", "to_out"]

# Verify model architecture matches
```

#### **4. Training Instability:**
```bash
# Reduce learning rate
--learning_rate 5e-5

# Increase warmup
--warmup_steps 2000

# Add gradient clipping
--gradient_clip_val 0.5
```

## 📋 Dataset Format

### **Video Files:**
- **Format**: MP4, AVI, MOV, MKV, WEBM
- **Resolution**: Minimum 224x224, recommended 512x512+
- **Duration**: Minimum 32 frames (1+ seconds at 30fps)
- **Quality**: Clean, watermark-free medical footage

### **Annotation Format:**
```json
{
  "video_id": "video_001",
  "descriptions": [
    "clean colonoscopy examination showing healthy mucosa",
    "endoscopic view of colon with clear visualization"
  ],
  "medical_findings": {
    "polyps_detected": false,
    "tissue_quality": "excellent"
  },
  "technical_quality": {
    "resolution": "512x512",
    "artifacts": "none"
  }
}
```

### **Metadata Format:**
```json
{
  "dataset_name": "Clean Colonoscopy Dataset",
  "total_videos": 100,
  "videos": [
    {
      "id": "video_001",
      "filename": "video_001.mp4",
      "split": "train",
      "is_clean": true,
      "duration_frames": 120
    }
  ]
}
```

## 🎛️ Advanced Usage

### **Custom LoRA Targets:**
```python
# Target specific layers
lora_config = LoRAConfig(
    target_modules=[
        "to_q", "to_k", "to_v",        # Attention
        "temporal_attention.*.to_q",    # Temporal layers
        "cross_attention.*.to_out"      # Cross-attention
    ]
)
```

### **Multi-GPU Training:**
```bash
python train_lora.py \
    --gpus 4 \
    --batch_size 16 \
    --strategy ddp
```

### **Resume Training:**
```bash
python train_lora.py \
    --resume_from_checkpoint ./lora_checkpoints/last.ckpt
```

## 📚 Technical Details

### **LoRA Implementation:**
- **Low-rank matrices**: A (random init) and B (zero init)
- **Adaptation**: `output = original_output + (input @ A.T @ B.T) * scaling`
- **Trainable parameters**: Only A and B matrices (~1% of total)
- **Merge capability**: Can merge LoRA weights into base model

### **Memory Efficiency:**
- **Parameter reduction**: ~99% fewer trainable parameters
- **Storage efficiency**: LoRA weights are only ~100MB vs 10GB base model
- **Fast adaptation**: Quick fine-tuning for domain-specific tasks

### **Watermark Elimination Strategy:**
1. **Clean data training**: Only watermark-free descriptions
2. **Anti-watermark terms**: Explicit avoidance of problematic words
3. **Medical domain focus**: Specialized colonoscopy vocabulary
4. **Prompt engineering**: Clean, professional medical language

## 🤝 Contributing

Feel free to improve this LoRA fine-tuning pipeline:

1. **Add new target modules** for LoRA adaptation
2. **Implement advanced optimizers** (8-bit, gradient checkpointing)
3. **Add evaluation metrics** (FVD, LPIPS, medical quality scores)
4. **Create domain-specific prompts** for different medical procedures

## 📄 License

This LoRA fine-tuning code follows the same license as the main DynamiCrafter project.

## 🎉 Happy Fine-tuning!

Transform your DynamiCrafter model into a watermark-free, medical domain specialist with LoRA! 🏥✨
