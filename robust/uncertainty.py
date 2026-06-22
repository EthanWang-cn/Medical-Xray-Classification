# -*- coding: utf-8 -*-
"""
Uncertainty Estimation Module
Implements various uncertainty estimation methods:
1. Monte Carlo Dropout (MC Dropout)
2. Deep Ensemble Uncertainty
3. Bayesian Neural Network Approximation
4. Temperature Scaling Calibration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class UncertaintyEstimator:
    """
    Uncertainty Estimator
    
    Provides various uncertainty estimation methods to help evaluate model prediction confidence
    Critical in key applications such as medical imaging
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
    
    def mc_dropout(self, x: torch.Tensor, n_samples: int = 30, 
                   dropout_rate: Optional[float] = None) -> Dict:
        """
        Monte Carlo Dropout Uncertainty Estimation
        
        Estimate epistemic uncertainty using prediction variance
        by keeping Dropout enabled during inference and performing multiple forward passes
        
        Args:
            x: Input image [B, C, H, W]
            n_samples: Number of samples (more is more accurate but slower)
            dropout_rate: If specified, temporarily modify Dropout rate
            
        Returns:
            dict: Contains mean, variance, entropy and other uncertainty metrics
        """
        x = x.to(self.device)
        
        # Save original Dropout state
        original_dropout_states = []
        if dropout_rate is not None:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    original_dropout_states.append(module.p)
                    module.p = dropout_rate
        
        # Keep training mode to enable Dropout
        self.model.train()
        
        # Collect results from multiple forward passes
        probs_list = []
        logits_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)
                logits_list.append(logits)
        
        # Switch back to evaluation mode
        self.model.eval()
        
        # Restore original Dropout rate
        if dropout_rate is not None:
            idx = 0
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = original_dropout_states[idx]
                    idx += 1
        
        # Stack results [n_samples, B, num_classes]
        probs_stack = torch.stack(probs_list, dim=0)
        logits_stack = torch.stack(logits_list, dim=0)
        
        # Calculate statistics
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        var_probs = probs_stack.var(dim=0)
        
        # Calculate predictive entropy
        eps = 1e-7
        entropy = - (mean_probs * torch.log(mean_probs + eps) + 
                    (1 - mean_probs) * torch.log(1 - mean_probs + eps))
        
        # Calculate Coefficient of Variation
        cv_probs = std_probs / (mean_probs + eps)
        
        return {
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'var_probs': var_probs,
            'entropy': entropy,
            'cv_probs': cv_probs,
            'n_samples': n_samples,
            'uncertainty_type': 'epistemic (MC Dropout)',
            'all_probs': probs_stack
        }
    
    def temperature_scaling(self, val_loader, n_classes: int, 
                           max_iter: int = 100, lr: float = 0.01) -> float:
        """
        Temperature Scaling Calibration
        
        Learn a temperature parameter T to better calibrate prediction probabilities
        Solve the problem of model being overconfident or underconfident
        
        Args:
            val_loader: Validation set data loader
            n_classes: Number of classes
            max_iter: Maximum number of iterations
            lr: Learning rate
            
        Returns:
            float: Optimal temperature parameter
        """
        # Initialize temperature parameter
        temperature = nn.Parameter(torch.ones(1).to(self.device))
        
        # Collect validation set logits and labels
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(self.device)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Optimize temperature parameter
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()
        
        def eval():
            optimizer.zero_grad()
            scaled_logits = all_logits.to(self.device) / temperature
            loss = criterion(scaled_logits, all_labels.to(self.device).float())
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        optimal_temp = temperature.item()
        print(f"Temperature scaling calibration complete, optimal temperature: {optimal_temp:.4f}")
        
        return optimal_temp
    
    def get_uncertainty_summary(self, uncertainty_dict: Dict) -> Dict:
        """
        Extract summary statistics from uncertainty results
        
        Args:
            uncertainty_dict: Return result from mc_dropout
            
        Returns:
            dict: Uncertainty summary
        """
        std = uncertainty_dict['std_probs']
        entropy = uncertainty_dict['entropy']
        
        return {
            'mean_std': std.mean().item(),
            'max_std': std.max().item(),
            'min_std': std.min().item(),
            'mean_entropy': entropy.mean().item(),
            'max_entropy': entropy.max().item(),
            'uncertainty_type': uncertainty_dict['uncertainty_type']
        }
    
    def classify_uncertainty(self, uncertainty_dict: Dict, 
                            low_threshold: float = 0.1,
                            high_threshold: float = 0.2) -> List[str]:
        """
        Classify uncertainty levels (low/medium/high)
        
        Args:
            uncertainty_dict: Uncertainty estimation results
            low_threshold: Low uncertainty threshold
            high_threshold: High uncertainty threshold
            
        Returns:
            list: Uncertainty level for each sample
        """
        std = uncertainty_dict['std_probs']
        sample_uncertainty = std.mean(dim=1)  # Average uncertainty per sample
        
        levels = []
        for unc in sample_uncertainty:
            unc_val = unc.item()
            if unc_val < low_threshold:
                levels.append('low')
            elif unc_val < high_threshold:
                levels.append('medium')
            else:
                levels.append('high')
        
        return levels


class AleatoricEstimator:
    """
    Aleatoric Uncertainty Estimator
    
    Estimate uncertainty inherent in the data itself (e.g., noise, ambiguity)
    Implemented by learning the variance of predictions
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Model that outputs logits and log_variance
            device: Computing device
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
    
    def estimate(self, x: torch.Tensor) -> Dict:
        """
        Estimate aleatoric uncertainty
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            dict: Contains mean prediction and uncertainty
        """
        x = x.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            
            # Assume model outputs (mean, log_var)
            if isinstance(output, tuple):
                mean, log_var = output
            else:
                # If model only outputs one value, aleatoric uncertainty cannot be estimated
                raise ValueError("Model needs to output both mean and log variance")
        
        var = torch.exp(log_var)
        std = torch.sqrt(var)
        
        return {
            'mean': mean,
            'variance': var,
            'std': std,
            'log_variance': log_var,
            'uncertainty_type': 'aleatoric'
        }


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, 
                               n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE)
    
    Measure the calibration degree of model prediction probabilities
    Lower ECE means more reliable probability estimates
    
    Args:
        probs: Prediction probabilities [N, num_classes]
        labels: True labels [N, num_classes]
        n_bins: Number of bins
        
    Returns:
        float: ECE value
    """
    # For multi-label classification, flatten all classes
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()
    
    # Bin by probability
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in current bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs_flat > bin_lower) & (probs_flat <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            # Calculate average accuracy and average confidence within the bin
            avg_confidence = probs_flat[in_bin].mean()
            avg_accuracy = labels_flat[in_bin].float().mean()
            
            ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return ece.item()
