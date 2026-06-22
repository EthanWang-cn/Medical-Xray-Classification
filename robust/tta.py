# -*- coding: utf-8 -*-
"""
Test-Time Augmentation (TTA)
Apply multiple transformations to input images during inference, then ensemble prediction results
Improve model robustness to image variations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Optional, Callable


class TestTimeAugmentation:
    """
    Test-Time Augmentation Class
    
    Implementation Principle:
    1. Apply various augmentation transformations to the same test image
    2. Make predictions for each transformed image separately
    3. Average or vote on all prediction results
    
    Advantages:
    - Improves prediction accuracy (typically 0.5-2%)
    - Enhances robustness to image translation, rotation, and scaling
    - No need to retrain the model
    """
    
    def __init__(self, model: nn.Module, image_size: int = 224, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Classification model
            image_size: Image size
            device: Computing device
        """
        self.model = model
        self.image_size = image_size
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Define TTA transformation pipeline
        self.transforms = self._build_tta_transforms()
    
    def _build_tta_transforms(self) -> List[Callable]:
        """
        Build TTA transformation list
        
        Includes various common test-time augmentation strategies:
        - Original image
        - Horizontal flip
        - Vertical flip
        - Multi-scale cropping
        - Color fine-tuning
        """
        transforms_list = []
        
        # 1. Original image (center crop)
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.CenterCrop(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 2. Horizontal flip
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.CenterCrop(self.image_size, self.image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 3. Vertical flip
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.CenterCrop(self.image_size, self.image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 4. Top-left crop
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.Crop(0, 0, self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 5. Bottom-right crop
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.Crop(32, 32, self.image_size + 32, self.image_size + 32),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 6. Slight rotation + center crop
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.Rotate(limit=5, p=1.0, border_mode=0),
                A.CenterCrop(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        return transforms_list
    
    def add_custom_transform(self, transform: Callable):
        """
        Add custom TTA transformation
        
        Args:
            transform: Albumentations transformation composition
        """
        self.transforms.append(transform)
        print(f"Added custom transform, currently {len(self.transforms)} transforms total")
    
    def predict(self, images: List[np.ndarray], weights: Optional[List[float]] = None) -> Dict:
        """
        TTA prediction
        
        Args:
            images: List of input images (numpy arrays, HWC format, RGB)
            weights: Weights for each transform, None means equal weights
            
        Returns:
            dict: TTA ensemble prediction results
        """
        if weights is not None and len(weights) != len(self.transforms):
            raise ValueError(f"Number of weights ({len(weights)}) does not match number of transforms ({len(self.transforms)})")
        
        all_probs = []
        
        with torch.no_grad():
            for i, transform in enumerate(self.transforms):
                # Apply transformation
                augmented_batch = []
                for img in images:
                    augmented = transform(image=img)['image']
                    augmented_batch.append(augmented)
                
                # Stack into batch
                batch = torch.stack(augmented_batch).to(self.device)
                
                # Predict
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
        
        # Stack [n_transforms, B, num_classes]
        probs_stack = torch.stack(all_probs, dim=0)
        
        # Calculate ensemble prediction
        if weights is not None:
            weights_tensor = torch.tensor(weights, device=self.device).view(-1, 1, 1)
            mean_probs = (probs_stack * weights_tensor).sum(dim=0)
        else:
            mean_probs = probs_stack.mean(dim=0)
        
        # Calculate uncertainty (variance between transforms)
        std_probs = probs_stack.std(dim=0)
        
        return {
            'probs': mean_probs,
            'std_probs': std_probs,
            'num_transforms': len(self.transforms),
            'all_probs': probs_stack
        }
    
    def predict_single(self, image: np.ndarray, weights: Optional[List[float]] = None) -> Dict:
        """
        TTA prediction for single image
        
        Args:
            image: Input image (numpy array, HWC format, RGB)
            weights: Weights for each transform
            
        Returns:
            dict: TTA prediction results
        """
        return self.predict([image], weights)


class LightTTA:
    """
    Lightweight TTA (horizontal flip only)
    Suitable for scenarios with high speed requirements
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Classification model
            device: Computing device
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
    
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Lightweight TTA prediction (horizontal flip only)
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            dict: TTA prediction results
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # Original image prediction
            logits1 = self.model(x)
            probs1 = torch.sigmoid(logits1)
            
            # Horizontal flip prediction
            x_flip = torch.flip(x, dims=[3])  # Horizontal flip
            logits2 = self.model(x_flip)
            probs2 = torch.sigmoid(logits2)
        
        # Average
        mean_probs = (probs1 + probs2) / 2.0
        
        return {
            'probs': mean_probs,
            'probs_original': probs1,
            'probs_flipped': probs2,
            'num_transforms': 2
        }
