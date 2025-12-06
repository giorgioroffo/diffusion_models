#!/usr/bin/env python3
"""
LoRA Wrapper for DynamiCrafter Fine-tuning
Implements LoRA (Low-Rank Adaptation) for efficient fine-tuning of DynamiCrafter models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for Linear modules.
    Adds trainable low-rank matrices to frozen pre-trained weights.
    """
    def __init__(
        self, 
        original_linear: nn.Linear, 
        rank: int = 16, 
        alpha: float = 32.0, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.original_linear.requires_grad_(False)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, original_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_linear.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output (frozen)
        original_out = self.original_linear(x)
        
        # LoRA adaptation
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return original_out + lora_out
    
    def merge_weights(self):
        """Merge LoRA weights into original linear layer (for inference)"""
        with torch.no_grad():
            delta_w = self.lora_B @ self.lora_A * self.scaling
            self.original_linear.weight.data += delta_w
            
    def reset_parameters(self):
        """Reset LoRA parameters"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    def __init__(
        self,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: List[str] = None,
        exclude_modules: List[str] = None
    ):
        self.rank = rank
        self.alpha = alpha  
        self.dropout = dropout
        self.target_modules = target_modules or [
            "to_q", "to_k", "to_v", "to_out",  # Cross-attention layers
            "to_k_ip", "to_v_ip",              # Image cross-attention
            "proj_in", "proj_out",             # Projection layers
        ]
        self.exclude_modules = exclude_modules or []


def apply_lora_to_module(module: nn.Module, config: LoRAConfig, prefix: str = "") -> int:
    """
    Recursively apply LoRA to target modules in a PyTorch model.
    
    Args:
        module: PyTorch module to modify
        config: LoRA configuration
        prefix: Current module path prefix
        
    Returns:
        Number of LoRA layers added
    """
    lora_count = 0
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Check if this is a target module
        is_target = any(target in full_name for target in config.target_modules)
        is_excluded = any(exclude in full_name for exclude in config.exclude_modules)
        
        if isinstance(child, nn.Linear) and is_target and not is_excluded:
            # Replace Linear layer with LoRA version
            lora_layer = LoRALinear(
                child, 
                rank=config.rank, 
                alpha=config.alpha, 
                dropout=config.dropout
            )
            setattr(module, name, lora_layer)
            lora_count += 1
            print(f"✅ Applied LoRA to: {full_name} (in_features={child.in_features}, out_features={child.out_features})")
        else:
            # Recursively process child modules
            lora_count += apply_lora_to_module(child, config, full_name)
    
    return lora_count


def wrap_dynamicrafter_with_lora(model, config: LoRAConfig) -> int:
    """
    Apply LoRA to DynamiCrafter model, focusing on UNet diffusion model.
    
    Args:
        model: DynamiCrafter model (LatentVisualDiffusion)
        config: LoRA configuration
        
    Returns:
        Number of LoRA layers applied
    """
    print("🚀 Applying LoRA to DynamiCrafter model...")
    print(f"📋 Target modules: {config.target_modules}")
    print(f"⚙️  LoRA config: rank={config.rank}, alpha={config.alpha}, dropout={config.dropout}")
    
    # Apply LoRA to the UNet diffusion model (main target)
    unet = model.model.diffusion_model
    lora_count = apply_lora_to_module(unet, config, "diffusion_model")
    
    # Optionally apply to conditioning encoders if needed
    if hasattr(model, 'cond_stage_model') and model.cond_stage_model is not None:
        lora_count += apply_lora_to_module(model.cond_stage_model, config, "cond_stage_model")
    
    # Apply to image conditioning if present
    if hasattr(model, 'embedder') and model.embedder is not None:
        lora_count += apply_lora_to_module(model.embedder, config, "embedder")
    
    print(f"✅ Applied LoRA to {lora_count} linear layers")
    return lora_count


def get_lora_parameters(model) -> List[torch.nn.Parameter]:
    """
    Get all LoRA parameters for optimization.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def freeze_non_lora_parameters(model):
    """
    Freeze all non-LoRA parameters in the model.
    
    Args:
        model: Model with LoRA layers
    """
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    print(f"🔒 Frozen parameters: {frozen_count}")
    print(f"🔓 Trainable LoRA parameters: {trainable_count}")
    print(f"📊 Trainable ratio: {trainable_count/(frozen_count+trainable_count)*100:.2f}%")


def save_lora_weights(model, save_path: str):
    """
    Save only LoRA weights to disk.
    
    Args:
        model: Model with LoRA layers
        save_path: Path to save LoRA weights
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    
    torch.save({
        'lora_state_dict': lora_state_dict,
        'config': {
            'rank': getattr(module, 'rank', 16),
            'alpha': getattr(module, 'alpha', 32.0),
            'scaling': getattr(module, 'scaling', 2.0),
        }
    }, save_path)
    print(f"💾 Saved LoRA weights to: {save_path}")


def load_lora_weights(model, load_path: str):
    """
    Load LoRA weights from disk.
    
    Args:
        model: Model with LoRA layers
        load_path: Path to load LoRA weights from
    """
    checkpoint = torch.load(load_path, map_location='cpu')
    lora_state_dict = checkpoint['lora_state_dict']
    
    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"
            
            if lora_a_key in lora_state_dict and lora_b_key in lora_state_dict:
                module.lora_A.data = lora_state_dict[lora_a_key].to(module.lora_A.device)
                module.lora_B.data = lora_state_dict[lora_b_key].to(module.lora_B.device)
                loaded_count += 1
    
    print(f"📥 Loaded LoRA weights from: {load_path}")
    print(f"✅ Loaded {loaded_count} LoRA layer pairs")


def print_lora_info(model):
    """Print information about LoRA layers in the model"""
    lora_layers = []
    total_params = 0
    lora_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_layers.append(name)
            params_a = module.lora_A.numel()
            params_b = module.lora_B.numel()
            lora_params += params_a + params_b
        
        for param in module.parameters(recurse=False):
            total_params += param.numel()
    
    print(f"\n📊 LoRA Model Information:")
    print(f"🔗 LoRA layers: {len(lora_layers)}")
    print(f"📈 Total parameters: {total_params:,}")
    print(f"🎯 LoRA parameters: {lora_params:,}")
    print(f"💡 LoRA ratio: {lora_params/total_params*100:.4f}%")
    print(f"📝 LoRA layers:")
    for layer in lora_layers[:10]:  # Show first 10
        print(f"   - {layer}")
    if len(lora_layers) > 10:
        print(f"   ... and {len(lora_layers)-10} more")
