# -*- coding: utf-8 -*-
"""
Deep Ensembles
Improve prediction accuracy and robustness by voting across multiple independently trained models
Also provides uncertainty estimation based on inter-model disagreement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class DeepEnsemble:
    """
    Deep Ensemble Learning Class
    
    Implementation Principle:
    1. Train multiple models with different initializations or data augmentations
    2. Average or weight predictions from all models during inference
    3. Estimate uncertainty through degree of disagreement between models
    
    Advantages:
    - Significantly improves classification accuracy
    - Provides reliable uncertainty estimates
    - More robust to noise and perturbations
    """
    
    def __init__(self, model_fn, num_models=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_fn: Model creation function, returns a new model instance when called
            num_models: Number of ensemble models
            device: Computing device
        """
        self.model_fn = model_fn
        self.num_models = num_models
        self.device = device
        
        # Model list
        self.models: List[nn.Module] = []
        
        # Model weights (for weighted ensemble)
        self.weights: Optional[List[float]] = None
    
    def create_models(self):
        """Create all ensemble models (different initializations)"""
        self.models = []
        for i in range(self.num_models):
            model = self.model_fn()
            model = model.to(self.device)
            self.models.append(model)
        print(f"Created {self.num_models} ensemble models")
    
    def load_models(self, checkpoint_paths: List[str]):
        """
        Load models from checkpoints
        
        Args:
            checkpoint_paths: List of model weight file paths
        """
        self.models = []
        for path in checkpoint_paths:
            model = self.model_fn()
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            self.models.append(model)
        
        self.num_models = len(self.models)
        print(f"Successfully loaded {self.num_models} ensemble models")
    
    def set_weights(self, weights: List[float]):
        """
        Set weights for each model (for weighted averaging)
        
        Args:
            weights: List of weights, length should equal number of models
        """
        if len(weights) != self.num_models:
            raise ValueError(f"Number of weights ({len(weights)}) does not match number of models ({self.num_models})")
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
        print(f"Set model weights: {self.weights}")
    
    def predict(self, x: torch.Tensor, return_all: bool = False) -> Dict:
        """
        Ensemble prediction
        
        Args:
            x: Input image [B, C, H, W]
            return_all: Whether to return individual predictions from all models
            
        Returns:
            dict: Contains ensemble prediction results and uncertainty information
        """
        if not self.models:
            raise RuntimeError("Models not loaded! Please call create_models() or load_models() first")
        
        x = x.to(self.device)
        
        # Collect predictions from all models
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(x)
                probs = torch.sigmoid(logits)
                all_logits.append(logits)
                all_probs.append(probs)
        
        # Stack [num_models, B, num_classes]
        probs_stack = torch.stack(all_probs, dim=0)
        logits_stack = torch.stack(all_logits, dim=0)
        
        # Calculate ensemble prediction
        if self.weights is not None:
            # Weighted average
            weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)
            mean_probs = (probs_stack * weights_tensor).sum(dim=0)
            mean_logits = (logits_stack * weights_tensor).sum(dim=0)
        else:
            # Simple average
            mean_probs = probs_stack.mean(dim=0)
            mean_logits = logits_stack.mean(dim=0)
        
        # Calculate uncertainty metrics
        std_probs = probs_stack.std(dim=0)  # Standard deviation (inter-model disagreement)
        
        # Calculate predictive entropy
        eps = 1e-7
        entropy = - (mean_probs * torch.log(mean_probs + eps) + 
                    (1 - mean_probs) * torch.log(1 - mean_probs + eps))
        
        result = {
            'probs': mean_probs,
            'logits': mean_logits,
            'std_probs': std_probs,
            'entropy': entropy,
            'num_models': self.num_models,
            'uncertainty_type': 'ensemble disagreement'
        }
        
        if return_all:
            result['all_probs'] = probs_stack
            result['all_logits'] = logits_stack
        
        return result
    
    def get_uncertainty_levels(self, x: torch.Tensor, thresholds: Optional[List[float]] = None) -> Dict:
        """
        Get prediction uncertainty levels
        
        Args:
            x: Input image
            thresholds: Uncertainty thresholds, default [0.1, 0.2, 0.3]
            
        Returns:
            dict: Uncertainty level for each sample
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3]
        
        result = self.predict(x)
        std_probs = result['std_probs']
        
        # Calculate average uncertainty for each sample
        sample_uncertainty = std_probs.mean(dim=1)  # [B]
        
        # Classification
        levels = []
        for unc in sample_uncertainty:
            unc_val = unc.item()
            if unc_val < thresholds[0]:
                levels.append('low')
            elif unc_val < thresholds[1]:
                levels.append('medium')
            else:
                levels.append('high')
        
        return {
            'uncertainty_values': sample_uncertainty,
            'uncertainty_levels': levels,
            'thresholds': thresholds
        }
    
    def save_models(self, save_dir: str, prefix: str = 'ensemble_model'):
        """
        Save all model weights
        
        Args:
            save_dir: Save directory
            prefix: Filename prefix
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            path = save_path / f"{prefix}_{i}.pth"
            torch.save(model.state_dict(), path)
        
        print(f"Saved {self.num_models} models to {save_dir}")
