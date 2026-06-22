# -*- coding: utf-8 -*-
"""
Adversarial Attack and Robustness Testing Module
Implements various adversarial attack methods and noise perturbations to test model robustness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class AdversarialTester:
    """
    Adversarial Robustness Tester
    
    Implements various adversarial attack methods to evaluate model performance under perturbations
    Helps identify model vulnerabilities and guides robustness improvements
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
    
    def fgsm_attack(self, x: torch.Tensor, y: torch.Tensor, 
                    epsilon: float = 0.03) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FGSM (Fast Gradient Sign Method) Adversarial Attack
        
        A classic method for quickly generating adversarial examples
        Adds perturbation along the sign direction of the loss gradient
        
        Args:
            x: Input image [B, C, H, W]
            y: True labels [B, num_classes]
            epsilon: Perturbation strength (larger = stronger attack)
            
        Returns:
            tuple: (adversarial example, perturbation)
        """
        x = x.to(self.device)
        y = y.to(self.device).float()
        
        # Enable gradients
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self.model(x_adv)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        grad_sign = x_adv.grad.data.sign()
        perturbation = epsilon * grad_sign
        x_adv = x + perturbation
        
        # Clip to valid range
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach(), perturbation.detach()
    
    def pgd_attack(self, x: torch.Tensor, y: torch.Tensor,
                   epsilon: float = 0.03, alpha: float = 0.01,
                   steps: int = 10, random_start: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PGD (Projected Gradient Descent) Adversarial Attack
        
        Iterative adversarial attack, stronger than FGSM
        Standard method for evaluating model robustness
        
        Args:
            x: Input image [B, C, H, W]
            y: True labels [B, num_classes]
            epsilon: Maximum perturbation range
            alpha: Step size per iteration
            steps: Number of iterations
            random_start: Whether to use random initialization
            
        Returns:
            tuple: (adversarial example, perturbation)
        """
        x = x.to(self.device)
        y = y.to(self.device).float()
        
        x_adv = x.clone().detach()
        
        # Random initialization
        if random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            logits = self.model(x_adv)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial example
            grad_sign = x_adv.grad.data.sign()
            x_adv = x_adv.detach() + alpha * grad_sign
            
            # Project back into epsilon ball
            perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + perturbation, 0, 1)
        
        return x_adv.detach(), (x_adv - x).detach()
    
    def gaussian_noise(self, x: torch.Tensor, std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gaussian Noise Perturbation
        
        Add Gaussian noise to test model robustness against random noise
        
        Args:
            x: Input image [B, C, H, W]
            std: Noise standard deviation
            
        Returns:
            tuple: (noisy image, noise)
        """
        x = x.to(self.device)
        
        noise = torch.randn_like(x) * std
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, 0, 1)
        
        return x_noisy, noise
    
    def salt_pepper_noise(self, x: torch.Tensor, amount: float = 0.05,
                          salt_vs_pepper: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Salt-and-Pepper Noise Perturbation
        
        Simulate pixel corruption in images
        
        Args:
            x: Input image [B, C, H, W]
            amount: Proportion of noisy pixels
            salt_vs_pepper: Ratio of salt noise vs pepper noise
            
        Returns:
            tuple: (noisy image, noise mask)
        """
        x = x.to(self.device)
        
        x_noisy = x.clone()
        batch_size, channels, height, width = x.shape
        
        # Generate noise mask
        n_noisy = int(amount * height * width)
        
        for i in range(batch_size):
            for c in range(channels):
                # Salt noise (white pixels)
                n_salt = int(n_noisy * salt_vs_pepper)
                coords = torch.randint(0, height * width, (n_salt,), device=self.device)
                h_coords = coords // width
                w_coords = coords % width
                x_noisy[i, c, h_coords, w_coords] = 1.0
                
                # Pepper noise (black pixels)
                n_pepper = n_noisy - n_salt
                coords = torch.randint(0, height * width, (n_pepper,), device=self.device)
                h_coords = coords // width
                w_coords = coords % width
                x_noisy[i, c, h_coords, w_coords] = 0.0
        
        noise_mask = (x_noisy != x).float()
        
        return x_noisy, noise_mask
    
    def evaluate_robustness(self, x: torch.Tensor, y: torch.Tensor,
                           attack_type: str = 'fgsm', **kwargs) -> Dict:
        """
        Evaluate model robustness under specific attack
        
        Args:
            x: Input image
            y: True labels
            attack_type: Attack type ('fgsm', 'pgd', 'gaussian', 'salt_pepper')
            **kwargs: Attack parameters
            
        Returns:
            dict: Robustness evaluation results
        """
        x = x.to(self.device)
        y = y.to(self.device).float()
        
        # Clean prediction
        with torch.no_grad():
            logits_clean = self.model(x)
            probs_clean = torch.sigmoid(logits_clean)
        
        # Generate adversarial/perturbed examples
        if attack_type == 'fgsm':
            x_adv, perturbation = self.fgsm_attack(x, y, **kwargs)
        elif attack_type == 'pgd':
            x_adv, perturbation = self.pgd_attack(x, y, **kwargs)
        elif attack_type == 'gaussian':
            x_adv, perturbation = self.gaussian_noise(x, **kwargs)
        elif attack_type == 'salt_pepper':
            x_adv, perturbation = self.salt_pepper_noise(x, **kwargs)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        # Adversarial prediction
        with torch.no_grad():
            logits_adv = self.model(x_adv)
            probs_adv = torch.sigmoid(logits_adv)
        
        # Calculate performance drop
        # Use threshold 0.5 to calculate accuracy
        preds_clean = (probs_clean > 0.5).float()
        preds_adv = (probs_adv > 0.5).float()
        
        acc_clean = (preds_clean == y).float().mean().item()
        acc_adv = (preds_adv == y).float().mean().item()
        
        # Calculate prediction changes
        prediction_change = (preds_clean != preds_adv).float().mean().item()
        
        # Calculate probability changes
        prob_diff = (probs_clean - probs_adv).abs().mean().item()
        
        return {
            'accuracy_clean': acc_clean,
            'accuracy_adversarial': acc_adv,
            'accuracy_drop': acc_clean - acc_adv,
            'prediction_change_rate': prediction_change,
            'mean_probability_difference': prob_diff,
            'attack_type': attack_type,
            'x_adversarial': x_adv,
            'perturbation': perturbation,
            'probs_clean': probs_clean,
            'probs_adversarial': probs_adv
        }
    
    def robustness_benchmark(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Run complete robustness benchmark
        
        Test model performance under various attacks and noise types
        
        Args:
            x: Input image
            y: True labels
            
        Returns:
            dict: Complete robustness test report
        """
        results = {}
        
        # 1. FGSM attack
        results['fgsm_eps0.01'] = self.evaluate_robustness(
            x, y, attack_type='fgsm', epsilon=0.01
        )
        results['fgsm_eps0.03'] = self.evaluate_robustness(
            x, y, attack_type='fgsm', epsilon=0.03
        )
        
        # 2. PGD attack
        results['pgd_eps0.03'] = self.evaluate_robustness(
            x, y, attack_type='pgd', epsilon=0.03, steps=10
        )
        
        # 3. Gaussian noise
        results['gaussian_std0.05'] = self.evaluate_robustness(
            x, y, attack_type='gaussian', std=0.05
        )
        results['gaussian_std0.1'] = self.evaluate_robustness(
            x, y, attack_type='gaussian', std=0.1
        )
        
        # 4. Salt-and-pepper noise
        results['salt_pepper_0.05'] = self.evaluate_robustness(
            x, y, attack_type='salt_pepper', amount=0.05
        )
        
        # Generate summary
        summary = {
            'clean_accuracy': results['fgsm_eps0.01']['accuracy_clean'],
            'robustness_scores': {}
        }
        
        for name, res in results.items():
            summary['robustness_scores'][name] = {
                'accuracy': res['accuracy_adversarial'],
                'accuracy_drop': res['accuracy_drop'],
                'prediction_change': res['prediction_change_rate']
            }
        
        return {
            'summary': summary,
            'detailed_results': results
        }
