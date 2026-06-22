# -*- coding: utf-8 -*-
"""
Robust Classifier Module
Integrates multiple robustness techniques, providing a unified inference interface
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbone import get_backbone


class RobustClassifier(nn.Module):
    """
    Robust Medical Image Classifier
    
    Integrates multiple robustness enhancement techniques:
    1. Deep Ensembles
    2. Monte Carlo Dropout Uncertainty Estimation
    3. Test-Time Augmentation (TTA)
    4. Adversarial Perturbation Robustness Testing
    
    Provides a unified inference interface supporting multiple robustness mode switching
    """
    
    def __init__(self, model_name='resnet50', num_classes=14,
                 pretrained=True, dropout_rate=0.3, in_channels=3,
                 n_ensembles=5):
        """
        Args:
            model_name: Backbone network name
            num_classes: Number of classification classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
            in_channels: Number of input channels
            n_ensembles: Number of ensemble models (for Deep Ensembles)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.n_ensembles = n_ensembles
        self.dropout_rate = dropout_rate
        
        # Create single model (single model mode)
        self.model = get_backbone(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            in_channels=in_channels
        )
        
        # Deep ensemble model list (for Deep Ensembles)
        self.ensemble_models = nn.ModuleList()
        
        # Flag indicating whether ensemble models have been loaded
        self._ensemble_loaded = False
    
    def forward(self, x):
        """
        Standard forward pass (single model)
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        return self.model(x)
    
    def predict_proba(self, x, mode='standard', **kwargs):
        """
        Predict probabilities, supporting multiple robustness modes
        
        Args:
            x: Input image [B, C, H, W]
            mode: Inference mode
                - 'standard': Standard inference
                - 'mc_dropout': Monte Carlo Dropout uncertainty estimation
                - 'ensemble': Deep ensemble inference
                - 'tta': Test-Time Augmentation
            **kwargs: Additional parameters for each mode
            
        Returns:
            dict: Dictionary containing predicted probabilities and uncertainty information
        """
        if mode == 'standard':
            return self._standard_predict(x)
        elif mode == 'mc_dropout':
            n_samples = kwargs.get('n_samples', 30)
            return self._mc_dropout_predict(x, n_samples)
        elif mode == 'ensemble':
            return self._ensemble_predict(x)
        else:
            raise ValueError(f"Unsupported inference mode: {mode}")
    
    def _standard_predict(self, x):
        """Standard inference"""
        self.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)  # Use sigmoid for multi-label classification
        
        return {
            'probs': probs,
            'logits': logits,
            'uncertainty': None
        }
    
    def _mc_dropout_predict(self, x, n_samples=30):
        """
        Monte Carlo Dropout Uncertainty Estimation
        
        Estimate prediction uncertainty (epistemic uncertainty)
        through multiple forward passes with Dropout enabled
        
        Args:
            x: Input image [B, C, H, W]
            n_samples: Number of samples
            
        Returns:
            dict: Contains mean probability, variance, entropy and other uncertainty metrics
        """
        # Keep training mode to enable Dropout
        self.train()
        
        # Collect results from multiple forward passes
        probs_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)
        
        # Stack all sample results [n_samples, B, num_classes]
        probs_stack = torch.stack(probs_list, dim=0)
        
        # Calculate statistics
        mean_probs = probs_stack.mean(dim=0)       # Mean probability
        std_probs = probs_stack.std(dim=0)         # Standard deviation
        var_probs = probs_stack.var(dim=0)         # Variance
        
        # Calculate predictive entropy (another measure of uncertainty)
        # For multi-label classification, calculate independently for each class
        eps = 1e-7
        entropy = - (mean_probs * torch.log(mean_probs + eps) + 
                    (1 - mean_probs) * torch.log(1 - mean_probs + eps))
        
        # Switch back to evaluation mode
        self.eval()
        
        return {
            'probs': mean_probs,
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'var_probs': var_probs,
            'entropy': entropy,
            'n_samples': n_samples,
            'uncertainty_type': 'epistemic (MC Dropout)'
        }
    
    def load_ensemble(self, model_paths):
        """
        Load deep ensemble models
        
        Args:
            model_paths: List of model weight paths
        """
        if len(model_paths) != self.n_ensembles:
            print(f"Warning: Expected {self.n_ensembles} models, actually loaded {len(model_paths)}")
        
        self.ensemble_models = nn.ModuleList()
        
        for path in model_paths:
            # Create new model instance
            model = get_backbone(
                model_name=self.model.__class__.__name__.replace('Classifier', '').lower(),
                num_classes=self.num_classes,
                pretrained=False,
                dropout_rate=self.dropout_rate
            )
            # Load weights
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            self.ensemble_models.append(model)
        
        self._ensemble_loaded = True
        print(f"Successfully loaded {len(self.ensemble_models)} ensemble models")
    
    def _ensemble_predict(self, x):
        """
        Deep Ensemble Inference
        
        Vote through multiple independently trained models,
        while providing uncertainty estimation
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            dict: Ensemble prediction results and uncertainty
        """
        if not self._ensemble_loaded:
            raise RuntimeError(
                "Ensemble models not loaded! Please call load_ensemble() first to load model weights"
            )
        
        probs_list = []
        
        with torch.no_grad():
            for model in self.ensemble_models:
                model.eval()
                logits = model(x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)
        
        # Stack results [n_ensembles, B, num_classes]
        probs_stack = torch.stack(probs_list, dim=0)
        
        # Calculate statistics
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        
        # Calculate prediction diversity (inter-model disagreement)
        # Use entropy to measure ensemble uncertainty
        eps = 1e-7
        entropy = - (mean_probs * torch.log(mean_probs + eps) + 
                    (1 - mean_probs) * torch.log(1 - mean_probs + eps))
        
        return {
            'probs': mean_probs,
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'entropy': entropy,
            'n_ensembles': len(self.ensemble_models),
            'uncertainty_type': 'ensemble disagreement'
        }
    
    def get_uncertainty_summary(self, result_dict):
        """
        Extract uncertainty summary from prediction results
        
        Args:
            result_dict: Return result from predict_proba
            
        Returns:
            dict: Uncertainty summary statistics
        """
        if 'std_probs' not in result_dict:
            return {'uncertainty': 'not available'}
        
        std = result_dict['std_probs']
        entropy = result_dict.get('entropy', None)
        
        summary = {
            'mean_std': std.mean().item(),
            'max_std': std.max().item(),
            'min_std': std.min().item(),
        }
        
        if entropy is not None:
            summary.update({
                'mean_entropy': entropy.mean().item(),
                'max_entropy': entropy.max().item(),
            })
        
        return summary
