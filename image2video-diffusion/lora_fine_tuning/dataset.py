#!/usr/bin/env python3
"""
Colonoscopy Dataset Loader for DynamiCrafter LoRA Fine-tuning
Handles loading and preprocessing of colonoscopy videos for training.
"""

import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


class ColonoscopyVideoDataset(Dataset):
    """
    Dataset for loading colonoscopy videos and their corresponding text descriptions.
    Expected structure:
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
    """
    
    def __init__(
        self,
        data_root: str = "/ops/datasets/colonoscopy",
        video_length: int = 16,
        resolution: int = 256,
        frame_stride: int = 1,
        split: str = "train",  # "train", "val", "test"
        min_video_frames: int = 32,
        clean_prompts_only: bool = True
    ):
        self.data_root = Path(data_root)
        self.video_length = video_length
        self.resolution = resolution
        self.frame_stride = frame_stride
        self.split = split
        self.min_video_frames = min_video_frames
        self.clean_prompts_only = clean_prompts_only
        
        # Paths
        self.videos_dir = self.data_root / "videos"
        self.annotations_dir = self.data_root / "annotations"
        self.metadata_file = self.data_root / "metadata.json"
        
        # Load dataset
        self.video_list = self._load_video_list()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        print(f"📁 Loaded {len(self.video_list)} {split} videos from {data_root}")
    
    def _load_video_list(self) -> List[Dict]:
        """Load and filter video list based on split and quality criteria"""
        video_list = []
        
        # Check if metadata exists
        if not self.metadata_file.exists():
            print(f"⚠️  No metadata.json found, scanning directory...")
            return self._scan_directory()
        
        # Load metadata
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for video_info in metadata.get('videos', []):
            # Filter by split
            if video_info.get('split') != self.split:
                continue
                
            # Check video file exists
            video_path = self.videos_dir / video_info['filename']
            if not video_path.exists():
                continue
                
            # Check annotation exists
            annotation_path = self.annotations_dir / f"{video_info['id']}.json"
            if not annotation_path.exists():
                continue
                
            # Filter by quality criteria
            if self._meets_quality_criteria(video_info):
                video_list.append({
                    'video_path': video_path,
                    'annotation_path': annotation_path,
                    'video_info': video_info
                })
        
        return video_list
    
    def _scan_directory(self) -> List[Dict]:
        """Fallback: scan videos directory if no metadata exists"""
        video_list = []
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        for video_file in self.videos_dir.glob('*'):
            if video_file.suffix.lower() in video_extensions:
                # Try to find corresponding annotation
                annotation_file = self.annotations_dir / f"{video_file.stem}.json"
                
                if annotation_file.exists():
                    video_list.append({
                        'video_path': video_file,
                        'annotation_path': annotation_file,
                        'video_info': {'id': video_file.stem, 'filename': video_file.name}
                    })
        
        # Simple train/val split if no metadata
        if self.split == "train":
            return video_list[:int(0.8 * len(video_list))]
        elif self.split == "val":
            return video_list[int(0.8 * len(video_list)):]
        else:
            return video_list
    
    def _meets_quality_criteria(self, video_info: Dict) -> bool:
        """Check if video meets quality criteria for training"""
        # Check minimum duration
        duration = video_info.get('duration_frames', 0)
        if duration < self.min_video_frames:
            return False
            
        # Check if marked as clean (no watermarks)
        if self.clean_prompts_only and not video_info.get('is_clean', True):
            return False
            
        # Check resolution
        height = video_info.get('height', 0)
        width = video_info.get('width', 0)
        if height < 224 or width < 224:  # Minimum resolution
            return False
            
        return True
    
    def _load_annotation(self, annotation_path: Path) -> Dict:
        """Load annotation file for a video"""
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        return annotation
    
    def _load_video_frames(self, video_path: Path) -> List[np.ndarray]:
        """Load video frames using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Sample frames for training sequence"""
        total_frames = len(frames)
        
        # Calculate required frames with stride
        required_frames = (self.video_length - 1) * self.frame_stride + 1
        
        if total_frames < required_frames:
            # Repeat frames if video is too short
            frames = frames * (required_frames // total_frames + 1)
            total_frames = len(frames)
        
        # Random start position
        max_start = total_frames - required_frames
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
        
        # Sample frames with stride
        sampled_frames = []
        for i in range(self.video_length):
            frame_idx = start_idx + i * self.frame_stride
            frame_idx = min(frame_idx, total_frames - 1)
            sampled_frames.append(frames[frame_idx])
        
        return sampled_frames
    
    def _get_clean_prompts(self, annotation: Dict) -> List[str]:
        """Get clean text descriptions without watermark-related terms"""
        prompts = annotation.get('descriptions', [])
        
        if not self.clean_prompts_only:
            return prompts
        
        # Filter out prompts with watermark-related terms
        watermark_terms = {
            'shutterstock', 'watermark', 'getty', 'alamy', 
            'stock', 'preview', 'sample', 'demo', 'logo'
        }
        
        clean_prompts = []
        for prompt in prompts:
            prompt_lower = prompt.lower()
            if not any(term in prompt_lower for term in watermark_terms):
                clean_prompts.append(prompt)
        
        # Add generic medical descriptions if no clean prompts available
        if not clean_prompts:
            clean_prompts = [
                "colonoscopy examination showing intestinal mucosa",
                "endoscopic view of colon with natural tissue appearance", 
                "medical colonoscopy procedure with clear visualization",
                "clean colonoscopy footage showing mucosal surface"
            ]
        
        return clean_prompts
    
    def __len__(self) -> int:
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_info = self.video_list[idx]
        
        # Load video frames
        frames = self._load_video_frames(video_info['video_path'])
        
        # Sample frames for sequence
        sampled_frames = self._sample_frames(frames)
        
        # Load annotation
        annotation = self._load_annotation(video_info['annotation_path'])
        
        # Get clean prompts
        clean_prompts = self._get_clean_prompts(annotation)
        selected_prompt = random.choice(clean_prompts)
        
        # Convert frames to tensors
        video_tensor = torch.stack([
            self.transform(Image.fromarray(frame)) 
            for frame in sampled_frames
        ])  # Shape: [T, C, H, W]
        
        # Rearrange to match DynamiCrafter expected format
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        return {
            'video': video_tensor,
            'caption': selected_prompt,
            'video_id': video_info['video_info']['id'],
            'first_frame': video_tensor[:, 0],  # First frame for conditioning
        }


class ColonoscopyDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for colonoscopy dataset
    """
    
    def __init__(
        self,
        data_root: str = "/ops/datasets/colonoscopy",
        batch_size: int = 4,
        num_workers: int = 4,
        video_length: int = 16,
        resolution: int = 256,
        frame_stride: int = 1,
        **kwargs
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.video_length = video_length
        self.resolution = resolution
        self.frame_stride = frame_stride
        self.kwargs = kwargs
        
    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = ColonoscopyVideoDataset(
                data_root=self.data_root,
                video_length=self.video_length,
                resolution=self.resolution,
                frame_stride=self.frame_stride,
                split="train",
                **self.kwargs
            )
            
            self.val_dataset = ColonoscopyVideoDataset(
                data_root=self.data_root,
                video_length=self.video_length,
                resolution=self.resolution,
                frame_stride=self.frame_stride,
                split="val",
                **self.kwargs
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ColonoscopyVideoDataset(
                data_root=self.data_root,
                video_length=self.video_length,
                resolution=self.resolution,
                frame_stride=self.frame_stride,
                split="test",
                **self.kwargs
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


def create_dummy_dataset(data_root: str = "/ops/datasets/colonoscopy"):
    """
    Create a dummy dataset structure for testing.
    This function helps you understand the expected dataset format.
    """
    print(f"🛠️  Creating dummy dataset structure in {data_root}")
    
    data_path = Path(data_root)
    videos_dir = data_path / "videos"
    annotations_dir = data_path / "annotations"
    
    # Create directories
    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample metadata
    metadata = {
        "dataset_name": "Clean Colonoscopy Dataset",
        "description": "Watermark-free colonoscopy videos for DynamiCrafter fine-tuning",
        "total_videos": 100,
        "videos": []
    }
    
    # Create dummy video entries
    for i in range(10):
        video_id = f"video_{i:03d}"
        metadata["videos"].append({
            "id": video_id,
            "filename": f"{video_id}.mp4",
            "split": "train" if i < 8 else "val",
            "duration_frames": 120,
            "fps": 30,
            "width": 512,
            "height": 512,
            "is_clean": True,
            "has_polyps": i % 3 == 0,
            "quality_score": 0.9
        })
        
        # Create dummy annotation
        annotation = {
            "video_id": video_id,
            "descriptions": [
                "clean colonoscopy examination showing healthy mucosa",
                "endoscopic view of colon with clear visualization",
                "medical colonoscopy procedure without artifacts",
                f"colonoscopy video {i} with natural tissue appearance"
            ],
            "medical_findings": {
                "polyps_detected": i % 3 == 0,
                "polyp_count": 1 if i % 3 == 0 else 0,
                "tissue_quality": "excellent",
                "visual_clarity": "high"
            },
            "technical_quality": {
                "resolution": "512x512",
                "brightness": "optimal", 
                "contrast": "good",
                "motion_blur": "minimal",
                "artifacts": "none"
            }
        }
        
        # Save annotation
        with open(annotations_dir / f"{video_id}.json", 'w') as f:
            json.dump(annotation, f, indent=2)
    
    # Save metadata
    with open(data_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created dummy dataset structure with {len(metadata['videos'])} video entries")
    print(f"📁 Dataset structure:")
    print(f"   {data_root}/")
    print(f"   ├── videos/          # Place your .mp4 files here")
    print(f"   ├── annotations/     # JSON files with descriptions")
    print(f"   └── metadata.json    # Dataset metadata")
    print(f"\n💡 Next steps:")
    print(f"   1. Add your colonoscopy videos to {videos_dir}/")
    print(f"   2. Update annotations with proper descriptions")
    print(f"   3. Ensure all descriptions are watermark-free")


if __name__ == "__main__":
    # Create dummy dataset for testing
    create_dummy_dataset()
    
    # Test dataset loading
    dataset = ColonoscopyVideoDataset(
        data_root="/ops/datasets/colonoscopy",
        video_length=16,
        resolution=256
    )
    
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Video shape: {sample['video'].shape}")
        print(f"Caption: {sample['caption']}")
