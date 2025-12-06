#!/usr/bin/env python3
"""
DynamiCrafter Video Generator - YAML Configuration Based
Generate videos from images using DynamiCrafter with YAML config input
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def construct_inference_command(config):
    """Construct the inference command based on config"""
    
    # Model parameters based on size
    model_params = {
        "256": {"H": 256, "W": 256, "FS": 3},
        "512": {"H": 512, "W": 512, "FS": 8}, 
        "1024": {"H": 1024, "W": 1024, "FS": 24}
    }
    
    model_size = str(config['model']['size'])
    if model_size not in model_params:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    params = model_params[model_size]
    
    # Paths
    checkpoint_path = config['model']['checkpoint_path']
    config_path = config['model']['config_path']
    output_dir = config['output']['directory']
    
    # Create output directory
    create_output_dir(output_dir)
    
    # Generation parameters
    gen_params = config.get('generation', {})
    seed = gen_params.get('seed', 123)
    n_samples = gen_params.get('n_samples', 1)
    batch_size = gen_params.get('batch_size', 1)
    guidance_scale = gen_params.get('guidance_scale', 7.5)
    ddim_steps = gen_params.get('ddim_steps', 50)
    ddim_eta = gen_params.get('ddim_eta', 1.0)
    video_length = gen_params.get('video_length', 16)
    frame_stride = gen_params.get('frame_stride', params['FS'])
    
    # Advanced parameters
    advanced_params = config.get('advanced', {})
    perframe_ae = advanced_params.get('perframe_ae', False)
    timestep_spacing = advanced_params.get('timestep_spacing', 'uniform')
    guidance_rescale = advanced_params.get('guidance_rescale', 0.0)
    
    # Input
    image_path = config['input']['image_path']
    prompt = config['input']['prompt']
    
    # Create prompt directory and file (required by inference.py)
    prompt_dir = f"/tmp/dynamicrafter_prompt_{seed}"
    os.makedirs(prompt_dir, exist_ok=True)
    prompt_file = os.path.join(prompt_dir, "prompt.txt")
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    
    # Copy image to prompt directory
    import shutil
    image_name = os.path.basename(image_path)
    prompt_image_path = os.path.join(prompt_dir, image_name)
    shutil.copy2(image_path, prompt_image_path)
    
    # Construct command
    cmd = [
        "python3", "scripts/evaluation/inference.py",
        "--seed", str(seed),
        "--ckpt_path", checkpoint_path,
        "--config", config_path,
        "--savedir", output_dir,
        "--n_samples", str(n_samples),
        "--bs", str(batch_size),
        "--height", str(params['H']),
        "--width", str(params['W']),
        "--unconditional_guidance_scale", str(guidance_scale),
        "--ddim_steps", str(ddim_steps),
        "--ddim_eta", str(ddim_eta),
        "--prompt_dir", prompt_dir,
        "--text_input",
        "--video_length", str(video_length),
        "--frame_stride", str(frame_stride),
        "--timestep_spacing", timestep_spacing,
        "--guidance_rescale", str(guidance_rescale)
    ]
    
    # Add perframe_ae flag if enabled
    if perframe_ae:
        cmd.append("--perframe_ae")
    
    return cmd, prompt_dir

def run_inference(config_path):
    """Main function to run inference with config"""
    
    print("🚀 DynamiCrafter - YAML Config Mode")
    print("=" * 50)
    
    # Load configuration
    config = load_config(config_path)
    print(f"📄 Config loaded: {config_path}")
    
    # Display configuration
    print(f"🖼️  Image: {config['input']['image_path']}")
    print(f"💬 Prompt: {config['input']['prompt']}")
    print(f"🎯 Model: {config['model']['size']}x{config['model']['size']}")
    print(f"📁 Output: {config['output']['directory']}")
    custom_filename = config['output'].get('filename', 'generated_video.mp4')
    print(f"📄 Filename: {custom_filename}")
    print()
    
    # Construct command
    cmd, temp_dir = construct_inference_command(config)
    
    # Set environment for RTX 5090 compatibility
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        # Run inference
        print("🔄 Starting video generation...")
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Video generation completed!")
            output_dir = config['output']['directory']
            
            # Get custom filename from config
            custom_filename = config['output'].get('filename', 'generated_video.mp4')
            if not custom_filename.endswith('.mp4'):
                custom_filename += '.mp4'
            
            # Videos are now saved directly to main directory (no subfolders!)
            if os.path.exists(output_dir):
                videos = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
                if videos:
                    # Find the generated video (should be named like "filename_sample0.mp4")
                    generated_video = None
                    for video in videos:
                        if '_sample' in video:
                            generated_video = video
                            break
                    
                    if generated_video:
                        original_path = os.path.join(output_dir, generated_video)
                        custom_path = os.path.join(output_dir, custom_filename)
                        
                        # Rename to custom filename
                        import shutil
                        shutil.move(original_path, custom_path)
                        
                        print(f"📁 Video saved as: {custom_path}")
                        return True
                    else:
                        print("❌ No generated video found in output directory!")
                        return False
                else:
                    print("❌ No video files found in output directory!")
                    return False
            else:
                print("❌ Output directory not found!")
                return False
        else:
            print("❌ Video generation failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False
    
    finally:
        # Cleanup temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="DynamiCrafter Video Generator - YAML Configuration Based")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--conda-env", default="dynami5090", help="Conda environment name")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Check if we're in the right conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != args.conda_env:
        print(f"⚠️  Warning: Expected conda environment '{args.conda_env}', but currently in '{conda_env}'")
        print(f"💡 Run: conda activate {args.conda_env}")
    
    # Run inference
    success = run_inference(args.config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
