#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-to-Image Generation Tutorial
==================================

Author: Dr Giorgio Roffo
Year: 2025-2026

Purpose:
--------
This tutorial is created for educational purposes to teach students and researchers
how to generate high-quality images from text descriptions using diffusion models.
You'll learn how to use Stable Diffusion XL (SDXL) with a two-stage approach 
(base + refiner) for maximum image quality.

License:
--------
This is open source and free code. You are welcome to use, modify, and repost
this tutorial for educational and research purposes.

What you'll learn:
- How to load pre-trained diffusion models from HuggingFace
- How to generate images from text prompts
- How to use a two-stage model (base + refiner) for maximum quality
- Best practices for prompt engineering

The process is simple:
1. Load the AI model
2. Give it a text description
3. Generate the image
4. Save it to a file

Real-World Applications:
------------------------
For practical examples of diffusion models applied to real-world scenarios, particularly
in medical imaging and endoscopy, see the companion repositories:

- **endo-diffusion-models/**: Contains SARA (Strategic AI Research and Application) 
  implementation of EndoDora, an image-conditioned video diffusion model specifically 
  tailored for colonoscopy and endoscopic video generation. This demonstrates how 
  to generate realistic endoscopic videos from seed images.

- **image2video-diffusion/**: Contains DynamiCrafter implementation for transforming 
  static images into dynamic video sequences using text prompts. Includes examples 
  of medical image animation, such as colonoscopy image-to-video generation for 
  dataset augmentation.

These repositories provide complete working examples of diffusion models applied to
medical imaging use cases, including endoscopy video generation and medical dataset
augmentation.

Citation:
---------
If you use this tutorial in your research or find it helpful, please cite the 
foundational papers:

BibTeX Citations:
-----------------
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

Text Citations:
---------------
Roffo, G., Biffi, C., Salvagnini, P., & Cherubini, A. (2024). Feature Selection Gates 
with Gradient Routing for Endoscopic Image Computing. In International Conference on 
Medical Image Computing and Computer-Assisted Intervention (pp. 339-349). Springer.

Roffo, G. (2025). The Origin of Self-Attention: Pairwise Affinity Matrices in Feature 
Selection and the Emergence of Self-Attention. arXiv preprint arXiv:2507.14560.

Roffo, G. (2025). A Survey of Large Language Models: Foundations and Future Directions.

Additional References:
---------------------
For diffusion models and Stable Diffusion XL:
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.
  CVPR 2022.
- Podell, D., et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution 
  Image Synthesis. arXiv preprint arXiv:2307.01952.
"""

# Step 1: Import the libraries we need
# =====================================
# These libraries provide the tools to work with AI image generation models

from diffusers import DiffusionPipeline  # Main library for diffusion models
import torch  # PyTorch - needed for GPU operations
from pathlib import Path  # For handling file paths


# Step 2: Define the function to generate an image
# =================================================
# This function takes a text prompt and creates an image from it

def generate_image(prompt: str, output_path: str):
    """
    Generate a high-quality image from a text description.
    
    Args:
        prompt: The text description of what you want to generate
               Example: "A beautiful sunset over mountains"
        output_path: Where to save the generated image
                    Example: "outputs/my_image.png"
    """
    
    # Step 2.1: Load the base model
    # ------------------------------
    # The base model creates the initial image
    # We use Stable Diffusion XL (SDXL) which is great for high-quality images
    print("Loading the base model...")
    print("(This downloads the model the first time - it's about 6GB)")
    
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",  # Model name from HuggingFace
        dtype=torch.float16,  # Use half precision to save memory (faster, uses less VRAM)
        variant="fp16",  # Use the fp16 (half precision) version
        use_safetensors=True,  # Use safetensors format (safer than pickle)
    )
    
    # Move the model to GPU if available (much faster than CPU)
    # If you don't have a GPU, this will use CPU (but it will be very slow)
    base.to("cuda")
    print("Base model loaded!")
    
    # Step 2.2: Load the refiner model
    # ---------------------------------
    # The refiner takes the base image and adds more detail and quality
    # This two-stage process gives us the best quality images
    print("\nLoading the refiner model...")
    print("(This also downloads on first run - about 6GB)")
    
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",  # Refiner model name
        text_encoder_2=base.text_encoder_2,  # Share the text encoder (saves memory)
        vae=base.vae,  # Share the VAE (saves memory)
        dtype=torch.float16,  # Use half precision
        use_safetensors=True,  # Use safetensors format
        variant="fp16",  # Use fp16 version
    )
    
    # Move refiner to GPU
    refiner.to("cuda")
    print("Refiner model loaded!")
    
    # Step 2.3: Set up generation parameters
    # ---------------------------------------
    # These parameters control how the image is generated
    
    n_steps = 256  # Number of steps - more steps = better quality but slower
                  # 50 is a good balance for high quality
    
    high_noise_frac = 0.8  # Use 80% of steps on base model, 20% on refiner
                          # This is the recommended split for best results
    
    print(f"\nGeneration settings:")
    print(f"  - Total steps: {n_steps}")
    print(f"  - Base model steps: {int(n_steps * high_noise_frac)}")
    print(f"  - Refiner steps: {int(n_steps * (1 - high_noise_frac))}")
    print(f"  - Image size: 1024x1024 pixels")
    
    # Step 2.4: Generate the image - Stage 1 (Base Model)
    # ----------------------------------------------------
    # The base model creates the initial image with high noise
    # We stop at 80% completion (high_noise_frac) to leave room for refinement
    
    print(f"\nStage 1: Generating base image...")
    print(f"Prompt: '{prompt}'")
    print("(This may take 1-2 minutes on GPU, or 10+ minutes on CPU)")
    
    image = base(
        prompt=prompt,  # The text description
        num_inference_steps=n_steps,  # Number of steps
        denoising_end=high_noise_frac,  # Stop at 80% (leave noise for refiner)
        output_type="latent",  # Output as latent representation (not final image yet)
        height=1024,  # Image height in pixels (1024 = high quality)
        width=1024,  # Image width in pixels (1024 = high quality)
        guidance_scale=7.5,  # How closely to follow the prompt (7.5 is good)
    ).images
    
    print("Base image generated!")
    
    # Step 2.5: Generate the image - Stage 2 (Refiner)
    # -------------------------------------------------
    # The refiner takes the base image and adds fine details
    # It starts from where the base model left off (80%)
    
    print(f"\nStage 2: Refining image for maximum detail...")
    print("(This adds fine details and improves quality)")
    
    image = refiner(
        prompt=prompt,  # Same prompt
        num_inference_steps=n_steps,  # Same number of steps
        denoising_start=high_noise_frac,  # Start from 80% (where base left off)
        image=image,  # Pass the base image
        guidance_scale=7.5,  # Same guidance scale
    ).images[0]
    
    print("Image refined!")
    
    # Step 2.6: Save the generated image
    # ------------------------------------
    # Create the output directory if it doesn't exist
    # Then save the image as a PNG file
    
    output_file = Path(output_path)  # Convert to Path object
    output_file.parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    
    image.save(output_path)  # Save the image
    print(f"\n✓ Image saved to: {output_path}")
    
    # Step 2.7: Clean up memory
    # --------------------------
    # Delete the models from memory to free up GPU/CPU resources
    # This is important if you want to generate more images later
    
    del base, refiner  # Delete the models
    torch.cuda.empty_cache()  # Clear GPU cache if using GPU
    print("Memory cleaned up!")


# Step 3: Main execution
# ======================
# This is where the script actually runs when you execute it

if __name__ == "__main__":
    # Step 3.1: Define what you want to generate
    # ------------------------------------------
    # Write a detailed description of the image you want
    # More detail in the prompt = better results!
    # 
    # Tips for good prompts:
    # - Be specific: "A red sports car" is better than "A car"
    # - Add style: "photorealistic", "oil painting", "anime style"
    # - Add quality words: "highly detailed", "sharp focus", "8k resolution"
    # - Add lighting: "cinematic lighting", "golden hour", "dramatic shadows"
    
    prompt = "A realistic image of Times Square New York City, highly detailed, sharp focus, intricate details, professional photography, 8k resolution, cinematic lighting, neon signs, bustling city street"
    
    # Step 3.2: Define where to save the image
    # -----------------------------------------
    # Choose a filename and location for your generated image
    
    output_path = "outputs/gen_image.png"
    
    # Step 3.3: Generate the image!
    # --------------------------------
    # Call our function to create the image
    
    print("=" * 80)
    print("Text-to-Image Generation")
    print("=" * 80)
    print()
    
    generate_image(prompt, output_path)
    
    print()
    print("=" * 80)
    print("Done! Check your image at:", output_path)
    print("=" * 80)
