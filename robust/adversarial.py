# -*- coding: utf-8 -*-
"""
对抗攻击与鲁棒性测试模块
实现多种对抗攻击方法和噪声扰动，用于测试模型的鲁棒性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class AdversarialTester:
    """
    对抗鲁棒性测试器
    
    实现多种对抗攻击方法，评估模型在扰动下的性能
    帮助发现模型的脆弱点，指导鲁棒性改进
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: 分类模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
    
    def fgsm_attack(self, x: torch.Tensor, y: torch.Tensor, 
                    epsilon: float = 0.03) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FGSM (Fast Gradient Sign Method) 对抗攻击
        
        快速生成对抗样本的经典方法
        沿着损失梯度的符号方向添加扰动
        
        Args:
            x: 输入图像 [B, C, H, W]
            y: 真实标签 [B, num_classes]
            epsilon: 扰动强度（越大攻击越强）
            
        Returns:
            tuple: (对抗样本, 扰动)
        """
        x = x.to(self.device)
        y = y.to(self.device).float()
        
        # 启用梯度
        x_adv = x.clone().detach().requires_grad_(True)
        
        # 前向传播
        logits = self.model(x_adv)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        grad_sign = x_adv.grad.data.sign()
        perturbation = epsilon * grad_sign
        x_adv = x + perturbation
        
        # 裁剪到有效范围
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach(), perturbation.detach()
    
    def pgd_attack(self, x: torch.Tensor, y: torch.Tensor,
                   epsilon: float = 0.03, alpha: float = 0.01,
                   steps: int = 10, random_start: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PGD (Projected Gradient Descent) 对抗攻击
        
        迭代式的对抗攻击，比 FGSM 更强
        是评估模型鲁棒性的标准方法
        
        Args:
            x: 输入图像 [B, C, H, W]
            y: 真实标签 [B, num_classes]
            epsilon: 最大扰动范围
            alpha: 每步的步长
            steps: 迭代步数
            random_start: 是否随机初始化
            
        Returns:
            tuple: (对抗样本, 扰动)
        """
        x = x.to(self.device)
        y = y.to(self.device).float()
        
        x_adv = x.clone().detach()
        
        # 随机初始化
        if random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(steps):
            x_adv.requires_grad_(True)
            
            # 前向传播
            logits = self.model(x_adv)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 更新对抗样本
            grad_sign = x_adv.grad.data.sign()
            x_adv = x_adv.detach() + alpha * grad_sign
            
            # 投影到 epsilon 球内
            perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + perturbation, 0, 1)
        
        return x_adv.detach(), (x_adv - x).detach()
    
    def gaussian_noise(self, x: torch.Tensor, std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        高斯噪声扰动
        
        添加高斯噪声，测试模型对随机噪声的鲁棒性
        
        Args:
            x: 输入图像 [B, C, H, W]
            std: 噪声标准差
            
        Returns:
            tuple: (带噪声图像, 噪声)
        """
        x = x.to(self.device)
        
        noise = torch.randn_like(x) * std
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, 0, 1)
        
        return x_noisy, noise
    
    def salt_pepper_noise(self, x: torch.Tensor, amount: float = 0.05,
                          salt_vs_pepper: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        椒盐噪声扰动
        
        模拟图像中的像素损坏
        
        Args:
            x: 输入图像 [B, C, H, W]
            amount: 噪声像素比例
            salt_vs_pepper: 盐噪声 vs 椒噪声比例
            
        Returns:
            tuple: (带噪声图像, 噪声掩码)
        """
        x = x.to(self.device)
        
        x_noisy = x.clone()
        batch_size, channels, height, width = x.shape
        
        # 生成噪声掩码
        n_noisy = int(amount * height * width)
        
        for i in range(batch_size):
            for c in range(channels):
                # 盐噪声（白色像素）
                n_salt = int(n_noisy * salt_vs_pepper)
                coords = torch.randint(0, height * width, (n_salt,), device=self.device)
                h_coords = coords // width
                w_coords = coords % width
                x_noisy[i, c, h_coords, w_coords] = 1.0
                
                # 椒噪声（黑色像素）
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
        评估模型在特定攻击下的鲁棒性
        
        Args:
            x: 输入图像
            y: 真实标签
            attack_type: 攻击类型 ('fgsm', 'pgd', 'gaussian', 'salt_pepper')
            **kwargs: 攻击参数
            
        Returns:
            dict: 鲁棒性评估结果
        """
        x = x.to(self.device)
        y = y.to(self.device).float()
        
        # 原始预测
        with torch.no_grad():
            logits_clean = self.model(x)
            probs_clean = torch.sigmoid(logits_clean)
        
        # 生成对抗/扰动样本
        if attack_type == 'fgsm':
            x_adv, perturbation = self.fgsm_attack(x, y, **kwargs)
        elif attack_type == 'pgd':
            x_adv, perturbation = self.pgd_attack(x, y, **kwargs)
        elif attack_type == 'gaussian':
            x_adv, perturbation = self.gaussian_noise(x, **kwargs)
        elif attack_type == 'salt_pepper':
            x_adv, perturbation = self.salt_pepper_noise(x, **kwargs)
        else:
            raise ValueError(f"不支持的攻击类型: {attack_type}")
        
        # 对抗样本预测
        with torch.no_grad():
            logits_adv = self.model(x_adv)
            probs_adv = torch.sigmoid(logits_adv)
        
        # 计算性能下降
        # 使用阈值 0.5 计算准确率
        preds_clean = (probs_clean > 0.5).float()
        preds_adv = (probs_adv > 0.5).float()
        
        acc_clean = (preds_clean == y).float().mean().item()
        acc_adv = (preds_adv == y).float().mean().item()
        
        # 计算预测变化
        prediction_change = (preds_clean != preds_adv).float().mean().item()
        
        # 计算概率变化
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
        运行完整的鲁棒性基准测试
        
        测试多种攻击和噪声下的模型性能
        
        Args:
            x: 输入图像
            y: 真实标签
            
        Returns:
            dict: 完整的鲁棒性测试报告
        """
        results = {}
        
        # 1. FGSM 攻击
        results['fgsm_eps0.01'] = self.evaluate_robustness(
            x, y, attack_type='fgsm', epsilon=0.01
        )
        results['fgsm_eps0.03'] = self.evaluate_robustness(
            x, y, attack_type='fgsm', epsilon=0.03
        )
        
        # 2. PGD 攻击
        results['pgd_eps0.03'] = self.evaluate_robustness(
            x, y, attack_type='pgd', epsilon=0.03, steps=10
        )
        
        # 3. 高斯噪声
        results['gaussian_std0.05'] = self.evaluate_robustness(
            x, y, attack_type='gaussian', std=0.05
        )
        results['gaussian_std0.1'] = self.evaluate_robustness(
            x, y, attack_type='gaussian', std=0.1
        )
        
        # 4. 椒盐噪声
        results['salt_pepper_0.05'] = self.evaluate_robustness(
            x, y, attack_type='salt_pepper', amount=0.05
        )
        
        # 生成摘要
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
