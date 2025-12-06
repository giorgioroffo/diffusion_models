#!/usr/bin/env python3
"""
DynamiCrafter LoRA Inference Script
Generate videos using a LoRA fine-tuned DynamiCrafter model
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

# Add parent directory to path to import DynamiCrafter modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import instantiate_from_config
from lora_wrapper import LoRAConfig, wrap_dynamicrafter_with_lora, load_lora_weights
from scripts.evaluation.inference import image_guided_synthesis
from utils.save_video import tensor_to_mp4


class LoRAInferenceModel:
    """
    Wrapper class for DynamiCrafter model with LoRA fine-tuning for inference
    """
    
    def __init__(
        self,
        base_model_path: str,
        lora_checkpoint_path: str,
        config_path: str,
        device: str = "cuda"
    ):
        self.device = device
        
        # Load model configuration
        print(f"📋 Loading model config from: {config_path}")
        config = OmegaConf.load(config_path)
        model_config = config.model
        
        # Load base model
        print(f"🚀 Loading base DynamiCrafter model...")
        self.model = instantiate_from_config(model_config)
        
        # Load base model weights
        if os.path.exists(base_model_path):
            print(f"📥 Loading base model weights from: {base_model_path}")
            checkpoint = torch.load(base_model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle key mismatches
            new_state_dict = {}
            for k, v in state_dict.items():
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
                print("✅ Base model weights loaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: {e}")
        
        # Apply LoRA modifications
        lora_config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["to_q", "to_k", "to_v", "to_out", "to_k_ip", "to_v_ip", "proj_in", "proj_out"]
        )
        
        print("🔧 Applying LoRA modifications...")
        wrap_dynamicrafter_with_lora(self.model, lora_config)
        
        # Load LoRA weights
        if os.path.exists(lora_checkpoint_path):
            print(f"📥 Loading LoRA weights from: {lora_checkpoint_path}")
            load_lora_weights(self.model, lora_checkpoint_path)
        else:
            print(f"⚠️  Warning: LoRA checkpoint not found at {lora_checkpoint_path}")
        
        # Move to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()
        
        print("✅ LoRA model ready for inference")
    
    def generate_video(
        self,
        image_path: str,
        prompt: str,
        video_length: int = 16,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        frame_stride: int = 3,
        eta: float = 1.0,
        seed: int = None
    ) -> torch.Tensor:
        """
        Generate video from image and text prompt
        
        Args:
            image_path: Path to input image
            prompt: Text description for video generation
            video_length: Number of frames to generate
            height: Video height
            width: Video width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            frame_stride: Frame stride for motion control
            eta: DDIM eta parameter
            seed: Random seed for reproducibility
            
        Returns:
            Generated video tensor [B, C, T, H, W]
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((width, height))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Prepare noise shape
        noise_shape = [1, 4, video_length, height // 8, width // 8]  # Latent space
        
        print(f"🎬 Generating video:")
        print(f"   Prompt: {prompt}")
        print(f"   Image: {image_path}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Frames: {video_length}")
        print(f"   Steps: {num_inference_steps}")
        print(f"   Guidance: {guidance_scale}")
        
        with torch.no_grad():
            # Generate video using image-guided synthesis
            video_tensor = image_guided_synthesis(
                model=self.model,
                prompts=[prompt],
                videos=image_tensor.unsqueeze(0),  # Add time dimension
                noise_shape=noise_shape,
                n_samples=1,
                ddim_steps=num_inference_steps,
                ddim_eta=eta,
                unconditional_guidance_scale=guidance_scale,
                cfg_img=None,
                fs=frame_stride,
                text_input=True,
                multiple_cond_cfg=False,
                loop=False,
                interp=False,
                timestep_spacing='uniform',
                guidance_rescale=0.0
            )
        
        return video_tensor[0]  # Remove batch dimension
    
    def save_video(
        self,
        video_tensor: torch.Tensor,
        output_path: str,
        fps: int = 8
    ):
        """Save video tensor to file"""
        print(f"💾 Saving video to: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save video
        tensor_to_mp4(video_tensor, output_path, fps=fps)
        
        print(f"✅ Video saved successfully")


def main():
    parser = argparse.ArgumentParser(description="DynamiCrafter LoRA Inference")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, 
                       default="/ops/model_checkpoints/diffusion_models/dynamicrafter_model.ckpt",
                       help="Path to base DynamiCrafter checkpoint")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                       help="Path to LoRA fine-tuned checkpoint")
    parser.add_argument("--config", type=str, default="config_lora.yaml",
                       help="Path to model config file")
    
    # Input arguments
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="./generated_video.mp4",
                       help="Output video path")
    parser.add_argument("--fps", type=int, default=8,
                       help="Output video FPS")
    
    # Generation parameters
    parser.add_argument("--video_length", type=int, default=16,
                       help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=256,
                       help="Video height")
    parser.add_argument("--width", type=int, default=256,
                       help="Video width")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--frame_stride", type=int, default=3,
                       help="Frame stride for motion control")
    parser.add_argument("--eta", type=float, default=1.0,
                       help="DDIM eta parameter")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"❌ Error: Input image not found: {args.image}")
        return
    
    if not os.path.exists(args.lora_checkpoint):
        print(f"❌ Error: LoRA checkpoint not found: {args.lora_checkpoint}")
        return
    
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        return
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = "cpu"
    
    print("🚀 DynamiCrafter LoRA Inference")
    print("=" * 50)
    
    # Initialize model
    model = LoRAInferenceModel(
        base_model_path=args.base_model,
        lora_checkpoint_path=args.lora_checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Generate video
    video_tensor = model.generate_video(
        image_path=args.image,
        prompt=args.prompt,
        video_length=args.video_length,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        frame_stride=args.frame_stride,
        eta=args.eta,
        seed=args.seed
    )
    
    # Save video
    model.save_video(video_tensor, args.output, fps=args.fps)
    
    print(f"🎉 Video generation completed!")
    print(f"📁 Output: {args.output}")


if __name__ == "__main__":
    main()
