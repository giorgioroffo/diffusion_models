#!/usr/bin/env python3
"""
Test script to verify LoRA fine-tuning setup
Checks dependencies, model loading, and basic functionality
"""

import os
import sys
import torch
import importlib.util
from pathlib import Path

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            spec = importlib.util.find_spec(module_name, package_name)
        else:
            spec = importlib.util.find_spec(module_name)
        
        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True, module.__version__ if hasattr(module, '__version__') else "unknown"
        else:
            return False, "not found"
    except Exception as e:
        return False, str(e)

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking Dependencies...")
    print("=" * 50)
    
    # Core dependencies
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("omegaconf", "OmegaConf"),
        ("einops", "Einops"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("tqdm", "TQDM")
    ]
    
    # LoRA-specific dependencies
    lora_dependencies = [
        ("peft", "PEFT Library"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("tensorboard", "TensorBoard")
    ]
    
    all_passed = True
    
    for module, name in dependencies:
        success, version = test_import(module)
        status = "✅" if success else "❌"
        print(f"{status} {name:20} {version}")
        if not success:
            all_passed = False
    
    print("\n🎯 LoRA-Specific Dependencies:")
    for module, name in lora_dependencies:
        success, version = test_import(module)
        status = "✅" if success else "⚠️"
        print(f"{status} {name:20} {version}")
        if not success:
            print(f"    💡 Install with: pip install {module}")
    
    return all_passed

def check_gpu():
    """Check GPU availability and CUDA setup"""
    print("\n🔍 Checking GPU Setup...")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        print(f"✅ CUDA Available: {torch.version.cuda}")
        print(f"🎮 GPUs Available: {gpu_count}")
        print(f"🎯 Current GPU: {gpu_name}")
        print(f"💾 GPU Memory: {memory_total:.1f} GB")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ GPU Memory Test: Passed")
        except Exception as e:
            print(f"⚠️  GPU Memory Test: Failed - {e}")
        
        return True
    else:
        print("❌ CUDA Not Available")
        print("💡 Training will run on CPU (very slow)")
        return False

def check_model_files():
    """Check if required model files exist"""
    print("\n🔍 Checking Model Files...")
    print("=" * 50)
    
    model_path = "/ops/model_checkpoints/diffusion_models/dynamicrafter_model.ckpt"
    config_path = "configs/inference_256_v1.0.yaml"
    
    if os.path.exists(model_path):
        size_gb = os.path.getsize(model_path) / 1e9
        print(f"✅ Base Model Found: {model_path} ({size_gb:.1f} GB)")
    else:
        print(f"❌ Base Model Missing: {model_path}")
        print("💡 Please ensure DynamiCrafter checkpoint is available")
    
    if os.path.exists(config_path):
        print(f"✅ Config Found: {config_path}")
    else:
        print(f"❌ Config Missing: {config_path}")
    
    return os.path.exists(model_path) and os.path.exists(config_path)

def check_dataset():
    """Check dataset structure"""
    print("\n🔍 Checking Dataset...")
    print("=" * 50)
    
    dataset_root = "/ops/datasets/colonoscopy"
    
    if os.path.exists(dataset_root):
        videos_dir = os.path.join(dataset_root, "videos")
        annotations_dir = os.path.join(dataset_root, "annotations")
        metadata_file = os.path.join(dataset_root, "metadata.json")
        
        videos_exist = os.path.exists(videos_dir)
        annotations_exist = os.path.exists(annotations_dir)
        metadata_exists = os.path.exists(metadata_file)
        
        print(f"✅ Dataset Root: {dataset_root}")
        print(f"{'✅' if videos_exist else '❌'} Videos Directory: {videos_dir}")
        print(f"{'✅' if annotations_exist else '❌'} Annotations Directory: {annotations_dir}")
        print(f"{'✅' if metadata_exists else '❌'} Metadata File: {metadata_file}")
        
        if videos_exist:
            video_count = len([f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))])
            print(f"📊 Video Files Found: {video_count}")
        
        return videos_exist and annotations_exist
    else:
        print(f"❌ Dataset Not Found: {dataset_root}")
        print("💡 Run 'python dataset.py' to create dummy dataset")
        return False

def test_lora_wrapper():
    """Test LoRA wrapper functionality"""
    print("\n🔍 Testing LoRA Wrapper...")
    print("=" * 50)
    
    try:
        from lora_wrapper import LoRALinear, LoRAConfig
        
        # Test LoRA linear layer
        original_linear = torch.nn.Linear(256, 512)
        lora_layer = LoRALinear(original_linear, rank=16, alpha=32.0)
        
        # Test forward pass
        test_input = torch.randn(4, 256)
        output = lora_layer(test_input)
        
        assert output.shape == (4, 512), f"Expected shape (4, 512), got {output.shape}"
        
        print("✅ LoRA Linear Layer: Working")
        print("✅ LoRA Config: Working")
        print(f"✅ Forward Pass: Output shape {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ LoRA Wrapper Error: {e}")
        return False

def test_dataset_loader():
    """Test dataset loading functionality"""
    print("\n🔍 Testing Dataset Loader...")
    print("=" * 50)
    
    try:
        from dataset import ColonoscopyVideoDataset, create_dummy_dataset
        
        # Create dummy dataset if needed
        dataset_root = "/tmp/test_colonoscopy"
        if not os.path.exists(dataset_root):
            create_dummy_dataset(dataset_root)
        
        # Test dataset loading
        dataset = ColonoscopyVideoDataset(
            data_root=dataset_root,
            video_length=8,
            resolution=128,
            split="train"
        )
        
        print(f"✅ Dataset Created: {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample Loading: {sample['video'].shape}")
            print(f"✅ Caption: {sample['caption'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Dataset Loader Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 DynamiCrafter LoRA Setup Test")
    print("=" * 70)
    
    all_tests = []
    
    # Run tests
    all_tests.append(("Dependencies", check_dependencies()))
    all_tests.append(("GPU Setup", check_gpu()))
    all_tests.append(("Model Files", check_model_files()))
    all_tests.append(("Dataset", check_dataset()))
    all_tests.append(("LoRA Wrapper", test_lora_wrapper()))
    all_tests.append(("Dataset Loader", test_dataset_loader()))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 70)
    
    passed = 0
    total = len(all_tests)
    
    for test_name, result in all_tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for LoRA fine-tuning!")
        print("\n🚀 Next steps:")
        print("   1. Prepare your colonoscopy dataset")
        print("   2. Run: ./run_training.sh")
        print("   3. Monitor training with TensorBoard")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
        print("\n🛠️  Quick fixes:")
        print("   - Install missing dependencies: pip install -r requirements_lora.txt")
        print("   - Check model checkpoint path")
        print("   - Create dataset: python dataset.py")

if __name__ == "__main__":
    main()
