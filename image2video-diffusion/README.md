
# Image-to-Video Diffusion Model

**Maintained by:** Dr Giorgio Roffo, Head of AI at SARA (Strategic AI Research and Application Department)

## Overview

This repository implements a state-of-the-art **video diffusion model** that transforms static images into dynamic video sequences using text prompts. The model leverages advanced **latent diffusion techniques** specifically designed for temporal video generation.

### What is a Diffusion Model?

Diffusion models are a class of generative models that learn to reverse a gradual noise corruption process. They work by:
1. **Forward Process**: Gradually adding noise to training data until it becomes pure noise
2. **Reverse Process**: Learning to denoise step-by-step, generating new samples from noise
3. **Conditioning**: Guiding generation using text prompts, images, or other control signals

### Video Diffusion Architecture

This implementation uses a **latent video diffusion model** that:
- Operates in a compressed latent space for computational efficiency
- Incorporates **temporal attention mechanisms** to maintain consistency across frames
- Uses **3D U-Net architecture** with spatial and temporal convolutions
- Supports **text-conditional** and **image-conditional** video generation
- Includes **motion dynamics modeling** through learned temporal priors

The specific model implemented here is **DynamiCrafter**, which specializes in animating open-domain images with video diffusion priors, enabling realistic motion synthesis from single static images.

## Resources

- **Paper**: https://arxiv.org/abs/2310.12190
- **Project Page**: https://doubiiu.github.io/projects/DynamiCrafter/
- **Hugging Face**: https://huggingface.co/papers/2310.12190
- **Demo Video**: https://youtu.be/0NfmIsNAg-g
- **Live Demo**: https://huggingface.co/spaces/Doubiiu/DynamiCrafter
- **Interpolation Demo**: https://huggingface.co/spaces/Doubiiu/DynamiCrafter_interp_loop
- **Model Weights**: https://huggingface.co/Doubiiu


## What is DynamiCrafter?

DynamiCrafter is the specific implementation that **animates static images into dynamic videos** using the power of text prompts and pre-trained video diffusion models. Simply provide an image and describe the motion you want to see, and DynamiCrafter will generate a realistic video clip that brings your image to life.

### How It Works:
1. **Input Image**: Provide any static image as the starting frame
2. **Text Prompt**: Describe the desired motion or action (e.g., "waves crashing on the shore")
3. **AI Magic**: DynamiCrafter uses advanced video diffusion priors to generate realistic motion
4. **Output Video**: Get a smooth, coherent video clip (up to 16 frames) with natural motion

### Key Features:
- **Any Image**: Works with open-domain images (photos, artwork, drawings)
- **Text-Guided**: Control motion with natural language descriptions
- **Multiple Resolutions**: 256×256, 512×320, 1024×576 video generation
- **Frame Interpolation**: Generate smooth transitions between two frames
- **Looping Videos**: Create seamless video loops
- **Creative Applications**: Perfect for storytelling, art animation, and content creation

---

## 🏥 **GI Colonoscopy Dataset Augmentation**

Transform static colonoscopy images into dynamic video clips for dataset augmentation. These examples demonstrate how DynamiCrafter can generate realistic endoscopic navigation sequences from single medical images, providing additional training data for medical AI systems.

### **📱 Normal Colonoscopy Navigation** 
<table class="center">
  <tr>
    <td><b>Static Input Image</b></td>
    <td><b>Generated Video Clip</b></td>
  </tr>
  <tr>
  <td>
    <img src=assets/colonoscopy/colonoscopy_input.jpg width="300">
    <br><i>Normal mucosal tissue</i>
  </td>
  <td>
    <img src=assets/colonoscopy/colonoscopy_test_256.gif width="300">
    <br><i>Endoscopic navigation sequence</i>
  </td>
  </tr>
</table>

### **🔬 Polypoid Lesions Detection**
<table class="center">
  <tr>
    <td><b>Static Input Image</b></td>
    <td><b>Generated Video Clip</b></td>
  </tr>
  <tr>
  <td>
    <img src=assets/colonoscopy/polyp_input.jpg width="300">
    <br><i>Polypoid lesion visualization</i>
  </td>
  <td>
    <img src=assets/colonoscopy/polyp_256.gif width="300">
    <br><i>Dynamic polyp examination</i>
  </td>
  </tr>
</table>

**Applications for Medical AI:**
- **Dataset Augmentation**: Generate additional training frames from limited medical images
- **Simulation Training**: Create realistic endoscopic sequences for medical education
- **Motion Patterns**: Learn natural camera movement in GI procedures
- **Anomaly Detection**: Augment rare pathological cases for better model training

---

## 🧰 **Available Models**

| **Model** | **Resolution** | **GPU Memory** | **Inference Time** | **Download** |
|-----------|----------------|----------------|-------------------|--------------|
| DynamiCrafter1024 | 576×1024 | 18.3GB | 75s (A100) | [🤗 HF](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt) |
| DynamiCrafter512 | 320×512 | 12.8GB | 20s (A100) | [🤗 HF](https://huggingface.co/Doubiiu/DynamiCrafter_512/blob/main/model.ckpt) |
| DynamiCrafter256 | 256×256 | 11.9GB | 10s (A100) | [🤗 HF](https://huggingface.co/Doubiiu/DynamiCrafter/blob/main/model.ckpt) |
| DynamiCrafter512_interp | 320×512 | 12.8GB | 20s (A100) | [🤗 HF](https://huggingface.co/Doubiiu/DynamiCrafter_512_Interp/blob/main/model.ckpt) |

*RTX 4090 memory usage: 18.3GB (1024), 12.8GB (512), 11.9GB (256)*

---

## ⚙️ **Installation & Setup**

### **📋 Prerequisites**
- NVIDIA GPU with 12GB+ VRAM (RTX 3090/4090/5090 recommended)
- CUDA 12.1+ for RTX 5090 compatibility
- Python 3.8-3.10
- Miniconda/Anaconda

### **🐍 Step 1: Install Miniconda (if not already installed)**

**Linux/WSL:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

**macOS:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
source ~/.zshrc
```

### **🚀 Step 2: Create RTX 5090 Compatible Environment**

For RTX 5090 users, use the optimized setup:
```bash
# Create environment with Python 3.10 for RTX 5090 compatibility
conda create -n dynami5090 python=3.10 -y
conda activate dynami5090

# Install RTX 5090-compatible PyTorch with CUDA 12.8+
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# Install project dependencies
pip install -r requirements.txt
```

**For other GPUs (RTX 3090/4090):**
```bash
conda create -n dynamicrafter python=3.8.5
conda activate dynamicrafter
pip install -r requirements.txt
```

### **📁 Step 3: Download Model Weights**

Create the model directory and download weights:
```bash
mkdir -p /ops/model_checkpoints/diffusion_models/
cd /ops/model_checkpoints/diffusion_models/

# Download 256x256 model (recommended for testing)
wget https://huggingface.co/Doubiiu/DynamiCrafter/resolve/main/model.ckpt -O dynamicrafter_model.ckpt

# Optional: Download higher resolution models
# wget https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt -O dynamicrafter_512_model.ckpt
# wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt -O dynamicrafter_1024_model.ckpt
```

---

## 🎯 **Usage**

### **🖥️ Command Line Interface (Recommended)**

DynamiCrafter now includes a clean Python script that uses YAML configuration for easy customization:

**1. Configure your generation settings:**

Edit `dynamicrafter_config.yaml`:
```yaml
model:
  size: 256  # Model size: 256, 512, or 1024
  checkpoint_path: "/ops/model_checkpoints/diffusion_models/dynamicrafter_model.ckpt"
  config_path: "configs/inference_256_v1.0.yaml"

input:
  image_path: "prompts/256/art.png"
  prompt: "man fishing in a boat at sunset"

output:
  directory: "./gen_videos"

generation:
  seed: 123
  n_samples: 1
  batch_size: 1
  guidance_scale: 7.5
  ddim_steps: 50
  ddim_eta: 1.0
  video_length: 16
```

**2. Generate your video:**
```bash
# Activate environment
conda activate dynami5090  # or dynamicrafter for other GPUs

# Run generation
python dynamicrafter_generate.py dynamicrafter_config.yaml
```

**3. Results:**
```
🚀 DynamiCrafter - YAML Config Mode
==================================================
📄 Config loaded: dynamicrafter_config.yaml
🖼️  Image: prompts/256/art.png
💬 Prompt: man fishing in a boat at sunset
🎯 Model: 256x256
📁 Output: ./gen_videos

🔄 Starting video generation...
✅ Video generation completed!
📁 Check output: ./gen_videos
🎬 Generated videos:
   ./gen_videos/samples_separate/art_sample0.mp4
```

### **🎨 Creative Examples**

Try these example configurations:

**Nature Scene:**
```yaml
input:
  image_path: "prompts/256/bloom01.png"
  prompt: "flowers blooming in spring breeze"
```

**Portrait Animation:**
```yaml
input:
  image_path: "prompts/256/girl3.jpeg"
  prompt: "girl talking and blinking"
```

**Action Scene:**
```yaml
input:
  image_path: "prompts/256/robot01.png"
  prompt: "robot walking through destroyed city"
```

### **🎮 Interactive Gradio Interface**

For a user-friendly web interface:
```bash
# Standard image-to-video
python gradio_app.py --res 256

# Frame interpolation and looping
python gradio_app_interp_and_loop.py
```

---

## 🎬 **Advanced Applications**

### **📚 Storytelling Video Generation**
<table class="center">
  <tr>
    <td colspan="4"><img src=assets/application/storytellingvideo.gif width="250"></td>
  </tr>
</table>

### **🔄 Generative Frame Interpolation**
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Starting Frame</td>
        <td>Ending Frame</td>
        <td>Generated Video</td>
    </tr>
  <tr>
  <td>
    <img src=assets/application/smile_start.png width="250">
  </td>
  <td>
    <img src=assets/application/smile_end.png width="250">
  </td>
  <td>
    <img src=assets/application/smile.gif width="250">
  </td>
  </tr>
</table>

### **♻️ Looping Video Generation**
<table class="center">
  <tr>
  <td>
    <img src=assets/application/60.gif width="300">
  </td>
  <td>
    <img src=assets/application/35.gif width="300">
  </td>
  <td>
    <img src=assets/application/36.gif width="300">
  </td>
  </tr>
</table>

---

## 🛠️ **Troubleshooting**

### **RTX 5090 Compatibility Issues**
If you encounter CUDA kernel errors:
1. Ensure PyTorch 2.4+ with CUDA 12.8+ is installed
2. Use the `dynami5090` environment with Python 3.10
3. The codebase has been optimized to disable problematic xformers operations

### **Memory Issues**
- Use 256×256 model for GPUs with <16GB VRAM
- Reduce `batch_size` in config
- Set `perframe_ae=False` for lower memory usage

### **Performance Tips**
- Reduce `ddim_steps` for faster generation (quality trade-off)
- Use GPU memory optimization: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

---

## 📝 **Changelog**

- **[2024.12.04]**: 🔥 Code improvements and optimizations by Dr Giorgio Roffo
- **[2024.06.14]**: 🔥🔥 Release training code for interpolation
- **[2024.05.24]**: Release WebVid10M-motion annotations
- **[2024.05.05]**: Release training code
- **[2024.03.14]**: Release generative frame interpolation and looping video models (320x512)
- **[2024.02.05]**: Release high-resolution models (320x512 & 576x1024)
- **[2023.12.02]**: Launch the local Gradio demo
- **[2023.11.29]**: Release the main model at a resolution of 256x256

---

## 👥 **Community & Support**

### **🔧 Community Tools**
- **ComfyUI**: [DynamiCrafterWrapper](https://github.com/kijai/ComfyUI-DynamiCrafterWrapper) by [kijai](https://twitter.com/kijaidesign)
- **Docker**: [DynamiCrafter_docker](https://github.com/maximofn/DynamiCrafter_docker) by [maximofn](https://github.com/maximofn)

### **🎨 Crafter Family**
- [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter): High-quality video generation framework
- [ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter): Tuning-free high-resolution generation
- [TaleCrafter](https://github.com/AILab-CVC/TaleCrafter): Interactive story visualization
- [LongerCrafter](https://github.com/arthur-qiu/LongerCrafter): Longer video generation
- [StyleCrafter](https://gongyeliu.github.io/StyleCrafter.github.io/): Stylized video generation
- [ViewCrafter](https://github.com/Drexubery/ViewCrafter): Novel view synthesis

---

## 📚 **Related Work & Similar Papers**

DynamiCrafter builds upon and relates to several groundbreaking works in video generation and diffusion models:

| **Paper** | **Venue** | **Description** |
|-----------|-----------|-----------------|
| **DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors** | ECCV 2024 (Oral) | 🎯 **This work** - Animates static images using text prompts and video diffusion priors |
| [**MagicVideo: Efficient Video Generation with Latent Diffusion Models**](https://arxiv.org/abs/2211.11018) | CVPR 2023 | Image/video conditioned video diffusion for efficient high-quality generation |
| [**VideoCrafter: Open Diffusion Models for High-Quality Video Generation**](https://arxiv.org/abs/2310.19512) | CVPR 2023 Workshop | Large-scale video diffusion toolkit for high-quality video synthesis |
| [**Tune-A-Video: One-Shot Tuning of Image Diffusion for Text-to-Video Generation**](https://arxiv.org/abs/2212.11565) | CVPR 2023 | Adapts image diffusion models for temporal video generation with minimal training |
| [**Make-A-Video: Text-to-Video Generation without Text-Video Data**](https://arxiv.org/abs/2209.14792) | CVPR 2023 | Early powerful text-to-video system leveraging image priors without video training data |

### **🔬 Key Contributions & Differences:**
- **DynamiCrafter** excels at **image animation** with precise motion control via text prompts
- **MagicVideo** focuses on **efficient latent diffusion** for video generation
- **VideoCrafter** provides a comprehensive **open-source toolkit** for video diffusion
- **Tune-A-Video** enables **one-shot adaptation** of image models for video tasks
- **Make-A-Video** pioneered **text-to-video without video training data**

---

## 📄 **Citation**

If you find DynamiCrafter helpful for your research, please cite the original DynamiCrafter paper:

```bibtex
@article{xing2023dynamicrafter,
  title={DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors},
  author={Xing, Jinbo and Xia, Menghan and Zhang, Yong and Chen, Haoxin and Yu, Wangbo and Liu, Hanyuan and Wang, Xintao and Wong, Tien-Tsin and Shan, Ying},
  journal={arXiv preprint arXiv:2310.12190},
  year={2023}
}
```

### Foundational Research Citations

If you use this implementation for medical imaging applications or find it helpful for your research, please also consider citing the foundational work on attention mechanisms and feature selection that influenced modern transformer architectures:

**BibTeX Citations:**

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

**Text Citations:**

Roffo, G., Biffi, C., Salvagnini, P., & Cherubini, A. (2024). Feature Selection Gates with Gradient Routing for Endoscopic Image Computing. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 339-349). Springer.

Roffo, G. (2025). The Origin of Self-Attention: Pairwise Affinity Matrices in Feature Selection and the Emergence of Self-Attention. arXiv preprint arXiv:2507.14560.

Roffo, G. (2025). A Survey of Large Language Models: Foundations and Future Directions.

---

## 🙏 **Acknowledgements**

- Original DynamiCrafter team from CUHK and Tencent AI Lab
- [AK(@_akhaliq)](https://twitter.com/_akhaliq?lang=en) for Hugging Face demo setup
- [camenduru](https://twitter.com/camenduru) for Replicate & Colab demos
- [Xinliang](https://github.com/dailingx) for open source contributions
- **[Dr Giorgio Roffo](https://www.linkedin.com/in/giorgio-roffo/)**, Head of AI at SARA (Strategic AI Research and Application Department), for contributions to this implementation

---

## 📚 **Learning Material & Citation**

This repository contains **educational and learning materials** for image-to-video diffusion models. Everyone is welcome to use this code for **learning purposes**—students, researchers, practitioners, and anyone interested in understanding how diffusion models can animate static images into dynamic video sequences.

**For Research Use**: If you use this code or find it helpful in your research, we would be pleased if you cite the foundational papers that influenced this work. Please see the [Citation](#-citation) section above for the complete BibTeX and text citation formats.

**Open Science Philosophy**: This code is shared freely because we believe in open science—the pursuit of knowledge, collaboration, and advancing the field benefits everyone. Scientists and researchers share code because they are led by science, not by proprietary interests. We hope this material helps accelerate learning and progress in AI and video generation.

---

## 📢 **Disclaimer**

This project aims to positively impact AI-driven video generation. Users have the freedom to create videos using this tool but are expected to comply with local laws and use it responsibly. The developers assume no responsibility for potential misuse.

---

**🎬 Ready to bring your images to life? Install DynamiCrafter and start creating amazing videos today!**