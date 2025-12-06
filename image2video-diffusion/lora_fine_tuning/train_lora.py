#!/usr/bin/env python3
"""
DynamiCrafter LoRA Fine-tuning Training Script
Fine-tunes DynamiCrafter model using LoRA on clean colonoscopy data to eliminate watermarks.
"""

import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Add parent directory to path to import DynamiCrafter modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import instantiate_from_config
from lora_wrapper import LoRAConfig, wrap_dynamicrafter_with_lora, get_lora_parameters, freeze_non_lora_parameters
from dataset import ColonoscopyDataModule


class DynamiCrafterLoRATrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training DynamiCrafter with LoRA
    """
    
    def __init__(
        self,
        model_config: dict,
        lora_config: LoRAConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        checkpoint_path: str = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Load base DynamiCrafter model
        print("🚀 Loading DynamiCrafter base model...")
        self.model = instantiate_from_config(model_config)
        
        # Load pretrained checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📥 Loading pretrained weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle key mismatches (e.g., framestride_embed -> fps_embedding)
            new_state_dict = {}
            for k, v in state_dict.items():
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
                print("✅ Successfully loaded pretrained weights")
            except Exception as e:
                print(f"⚠️  Warning: Could not load some weights: {e}")
        
        # Apply LoRA to the model
        print("🔧 Applying LoRA modifications...")
        self.lora_layers_count = wrap_dynamicrafter_with_lora(self.model, lora_config)
        
        # Freeze non-LoRA parameters
        freeze_non_lora_parameters(self.model)
        
        print(f"✅ LoRA setup complete: {self.lora_layers_count} layers modified")
    
    def forward(self, batch):
        """Forward pass through the model"""
        video = batch['video']  # [B, C, T, H, W]
        caption = batch['caption']  # List of strings
        
        # DynamiCrafter expects specific input format
        # This is a simplified version - you may need to adapt based on actual model
        return self.model(video, caption)
    
    def training_step(self, batch, batch_idx):
        """Training step with diffusion loss"""
        video = batch['video']  # [B, C, T, H, W]
        caption = batch['caption']  # List of strings
        
        # Get first frame for conditioning
        first_frame = video[:, :, 0]  # [B, C, H, W]
        
        # Encode to latent space
        with torch.no_grad():
            # Encode video to latent space
            b, c, t, h, w = video.shape
            video_flat = video.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
            latents_flat = self.model.encode_first_stage(video_flat)
            _, c_latent, h_latent, w_latent = latents_flat.shape
            latents = latents_flat.reshape(b, t, c_latent, h_latent, w_latent).permute(0, 2, 1, 3, 4)
        
        # Sample timesteps
        timesteps = torch.randint(0, self.model.num_timesteps, (b,), device=self.device)
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.model.q_sample(latents, timesteps, noise=noise)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.model.get_learned_conditioning(caption)
            
            # Get image embeddings for conditioning
            img_embeddings = self.model.embedder(first_frame)
            
            # Combine text and image conditioning
            conditioning = torch.cat([text_embeddings, img_embeddings], dim=1)
        
        # Predict noise
        model_pred = self.model.apply_model(noisy_latents, timesteps, conditioning)
        
        # Calculate loss (MSE between predicted and actual noise)
        loss = F.mse_loss(model_pred, noise, reduction='mean')
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, logger=True, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)
        
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Get only LoRA parameters
        lora_params = get_lora_parameters(self.model)
        
        print(f"🎯 Optimizing {len(lora_params)} LoRA parameters")
        
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.max_steps // 4,
            T_mult=2,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint):
        """Save only LoRA parameters"""
        # Remove non-LoRA parameters from checkpoint to save space
        lora_state_dict = {}
        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_state_dict[name] = param.cpu()
        
        checkpoint['lora_state_dict'] = lora_state_dict
        return checkpoint


def load_model_config(config_path: str) -> dict:
    """Load model configuration from YAML file"""
    config = OmegaConf.load(config_path)
    return config.model


def main():
    parser = argparse.ArgumentParser(description="DynamiCrafter LoRA Fine-tuning")
    
    # Model arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to model config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to pretrained DynamiCrafter checkpoint")
    
    # Dataset arguments
    parser.add_argument("--data_root", type=str, default="/ops/datasets/colonoscopy",
                       help="Root directory of colonoscopy dataset")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--video_length", type=int, default=16,
                       help="Number of frames per video sequence")
    parser.add_argument("--resolution", type=int, default=256,
                       help="Video resolution (height and width)")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank (lower = more efficient)")
    parser.add_argument("--lora_alpha", type=float, default=32.0,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout rate")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for LoRA parameters")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                       help="Weight decay for regularization")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum number of training epochs")
    parser.add_argument("--max_steps", type=int, default=50000,
                       help="Maximum number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./lora_checkpoints",
                       help="Directory to save LoRA checkpoints")
    parser.add_argument("--experiment_name", type=str, default="dynamicrafter_lora",
                       help="Name for the experiment")
    
    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--precision", type=str, default="16-mixed",
                       help="Training precision (16-mixed, 32)")
    
    args = parser.parse_args()
    
    # Set up directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configuration
    print(f"📋 Loading model config from: {args.config}")
    model_config = load_model_config(args.config)
    
    # Create LoRA configuration
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out",  # Cross-attention
            "to_k_ip", "to_v_ip",              # Image cross-attention
            "proj_in", "proj_out",             # Projections
        ]
    )
    
    print(f"🎯 LoRA Configuration:")
    print(f"   Rank: {lora_config.rank}")
    print(f"   Alpha: {lora_config.alpha}")
    print(f"   Dropout: {lora_config.dropout}")
    print(f"   Target modules: {lora_config.target_modules}")
    
    # Create data module
    print(f"📁 Setting up data from: {args.data_root}")
    data_module = ColonoscopyDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        video_length=args.video_length,
        resolution=args.resolution,
        clean_prompts_only=True
    )
    
    # Create model
    print("🏗️  Creating LoRA model...")
    model = DynamiCrafterLoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        checkpoint_path=args.checkpoint
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename=f"{args.experiment_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True
        )
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.experiment_name,
        version=None
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.gpus,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=50,
        val_check_interval=0.25,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Start training
    print("🚀 Starting LoRA fine-tuning...")
    print(f"💾 Checkpoints will be saved to: {args.output_dir}")
    print(f"📊 TensorBoard logs: {logger.log_dir}")
    
    trainer.fit(model, data_module)
    
    print("✅ Training completed!")
    print(f"📁 Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()
