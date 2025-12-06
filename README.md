# Diffusion Models Tutorial

**Author**: Dr Giorgio Roffo  
**Year**: 2025-2026  
**Purpose**: Educational tutorial for learning diffusion models

## 📚 Overview

This repository provides a simple, educational tutorial on how to generate high-quality images from text descriptions using diffusion models. You'll learn how to use **Stable Diffusion XL (SDXL)** with a two-stage approach (base + refiner) for maximum image quality.

### What You'll Learn

- How to load pre-trained diffusion models from HuggingFace
- How to generate images from text prompts
- How to use a two-stage model (base + refiner) for maximum quality
- Best practices for prompt engineering

### The Process

1. Load the AI model
2. Give it a text description
3. Generate the image
4. Save it to a file

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or higher** (Python 3.10+ recommended)
- **NVIDIA GPU with CUDA support** (recommended, but CPU will work - just slower)
- At least **12GB GPU memory** (for GPU) or **16GB RAM** (for CPU)
- **~15GB free disk space** (for models and dependencies)
- Stable internet connection (for downloading models)

### Step 1: Clone the Repository

```bash
git clone git@github.com:giorgioroffo/diffusion_models.git
cd diffusion_models
```

### Step 2: Create a Python Virtual Environment

Create a new virtual environment to isolate dependencies:

```bash
# Create virtual environment named 'venv'
python3 -m venv venv
```

**Note**: On some systems, you may need to use `python` instead of `python3`. Check your Python version with `python3 --version` or `python --version`.

### Step 3: Activate the Virtual Environment

Activate the virtual environment before installing packages:

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the beginning of your command prompt, indicating the environment is active.

### Step 4: Upgrade pip (Recommended)

Upgrade pip to the latest version for better package management:

```bash
pip install --upgrade pip
```

### Step 5: Install Dependencies

Install all required packages with pinned versions:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.9.1 (with CUDA support if available)
- Diffusers 0.35.2
- Transformers 4.57.3
- Accelerate 1.12.0
- Pillow 12.0.0
- Safetensors 0.7.0
- And all their dependencies

**Installation time**: This may take 5-10 minutes depending on your internet connection, as PyTorch is a large package (~900MB).

### Step 6: Verify Installation

Verify that all packages are installed correctly:

```bash
python3 -c "import torch; import diffusers; import transformers; print('✓ All packages installed successfully!')"
```

If you see the success message, you're ready to proceed!

### Step 7: Run the Tutorial

Run the main script:

```bash
python diffusion_models.py
```

**First run**: The script will automatically download the Stable Diffusion XL models (~12GB total) from HuggingFace. This happens automatically and may take 10-30 minutes depending on your internet speed.

**What happens:**
1. The script loads the base model (~6GB)
2. The script loads the refiner model (~6GB, shares some components)
3. Generates an image based on the prompt in the script
4. Saves the result to `outputs/gen_image.png`

**Generation time:**
- **GPU**: 1-2 minutes per image
- **CPU**: 10+ minutes per image (not recommended)

### Step 8: Customize Your Prompt

Edit `diffusion_models.py` and change the `prompt` variable around line 257:

```python
prompt = "Your custom text description here"
```

**Tips for good prompts:**
- **Be specific**: "A red sports car" is better than "A car"
- **Add style**: "photorealistic", "oil painting", "anime style", "watercolor"
- **Add quality words**: "highly detailed", "sharp focus", "8k resolution", "professional photography"
- **Add lighting**: "cinematic lighting", "golden hour", "dramatic shadows", "soft natural light"
- **Add composition**: "wide angle", "close-up", "portrait", "landscape"

**Example prompts:**
```python
prompt = "A majestic lion standing on a rock at sunset, highly detailed, photorealistic, 8k resolution, cinematic lighting, golden hour, dramatic sky"
prompt = "A futuristic cityscape at night with neon lights, cyberpunk style, highly detailed, sharp focus, 4k resolution"
prompt = "A serene Japanese garden with cherry blossoms, watercolor style, soft pastel colors, peaceful atmosphere"
```

### Step 9: Deactivate Virtual Environment (When Done)

When you're finished working, you can deactivate the virtual environment:

```bash
deactivate
```

The `(venv)` prefix will disappear from your command prompt.

---

## 📁 Repository Structure

```
diffusion_models/
├── diffusion_models.py          # Main tutorial script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── outputs/                     # Generated images (created automatically)
├── endo-diffusion-models/       # Medical imaging examples (see below)
└── image2video-diffusion/       # Video generation examples (see below)
```

---

## 🏥 Real-World Applications

For practical examples of diffusion models applied to real-world scenarios, particularly in medical imaging and endoscopy, see the companion repositories:

### **endo-diffusion-models/**

Contains SARA (Strategic AI Research and Application) implementation of **EndoDora**, an image-conditioned video diffusion model specifically tailored for colonoscopy and endoscopic video generation. This demonstrates how to generate realistic endoscopic videos from seed images.

**Key Features:**
- Image-conditioned video generation for medical imaging
- Specialized for colonoscopy and endoscopy
- Complete working examples with pre-trained models

See `endo-diffusion-models/README.md` for detailed documentation.

### **image2video-diffusion/**

Contains **DynamiCrafter** implementation for transforming static images into dynamic video sequences using text prompts. Includes examples of medical image animation, such as colonoscopy image-to-video generation for dataset augmentation.

**Key Features:**
- Image-to-video generation with text prompts
- Medical image animation examples
- Multiple resolution support (256×256, 512×320, 1024×576)

See `image2video-diffusion/README.md` for detailed documentation.

These repositories provide complete working examples of diffusion models applied to medical imaging use cases, including endoscopy video generation and medical dataset augmentation.

---

## 📖 Learning Material & Citation

This repository contains **educational and learning materials** for diffusion models. Everyone is welcome to use this code for **learning purposes**—students, researchers, practitioners, and anyone interested in understanding how diffusion models work.

**For Research Use**: If you use this code or find it helpful in your research, we would be pleased if you cite the foundational papers. Please see the [Citation](#citation) section below for the complete BibTeX and text citation formats.

**Open Science Philosophy**: This code is shared freely because we believe in open science—the pursuit of knowledge, collaboration, and advancing the field benefits everyone. Scientists and researchers share code because they are led by science, not by proprietary interests.

---

## 📝 Citation

If you use this tutorial in your research or find it helpful, please cite the foundational papers:

### BibTeX Citations

```bibtex
@inproceedings{roffo2024feature,
  title = {Feature Selection Gates with Gradient Routing for Endoscopic Image Computing},
  author = {Roffo, Giorgio and Biffi, Carlo and Salvagnini, Pietro and Cherubini, Andrea},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages = {339--349},
  year = {2024},
  organization = {Springer}
}

@article{roffo2025origin,
  title = {The Origin of Self-Attention: Pairwise Affinity Matrices in Feature Selection and the Emergence of Self-Attention},
  author = {Roffo, Giorgio},
  journal = {arXiv preprint arXiv:2507.14560},
  year = {2025}
}

@article{roffo2025survey,
  title = {A Survey of Large Language Models: Foundations and Future Directions},
  author = {Roffo, Giorgio},
  year = {2025}
}
```

### Text Citations

Roffo, G., Biffi, C., Salvagnini, P., & Cherubini, A. (2024). Feature Selection Gates with Gradient Routing for Endoscopic Image Computing. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 339-349). Springer.

Roffo, G. (2025). The Origin of Self-Attention: Pairwise Affinity Matrices in Feature Selection and the Emergence of Self-Attention. arXiv preprint arXiv:2507.14560.

Roffo, G. (2025). A Survey of Large Language Models: Foundations and Future Directions.

### Additional References

For diffusion models and Stable Diffusion XL:
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
- Podell, D., et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv preprint arXiv:2307.01952.

---

## 🛠️ Troubleshooting

### Out of Memory Errors

If you encounter GPU memory errors:
- Reduce the image resolution in `diffusion_models.py` (change `height=1024, width=1024` to `height=512, width=512`)
- Reduce `n_steps` (currently 256) to a lower value like 50
- Use CPU instead (will be much slower)

### Model Download Issues

If the model download fails:
- Check your internet connection
- Ensure you have enough disk space (~12GB)
- The models are downloaded from HuggingFace, so ensure you can access huggingface.co

### CUDA/GPU Issues

If you don't have a GPU or want to use CPU:
- The code will automatically fall back to CPU if CUDA is not available
- Note: CPU generation is very slow (10+ minutes per image)

---

## 📄 License

This is open source and free code. You are welcome to use, modify, and repost this tutorial for educational and research purposes.

---

## 👤 Maintainer

**Dr Giorgio Roffo**

*Advancing AI through Open Science and Education*

