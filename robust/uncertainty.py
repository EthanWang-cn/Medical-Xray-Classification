# -*- coding: utf-8 -*-
"""
不确定性估计模块
实现多种不确定性估计方法：
1. 蒙特卡洛 Dropout (MC Dropout)
2. 深度集成不确定性
3. 贝叶斯神经网络近似
4. 温度缩放校准
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class UncertaintyEstimator:
    """
    不确定性估计器
    
    提供多种不确定性估计方法，帮助评估模型预测的可信度
    在医学影像等关键应用中，不确定性估计至关重要
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
    
    def mc_dropout(self, x: torch.Tensor, n_samples: int = 30, 
                   dropout_rate: Optional[float] = None) -> Dict:
        """
        蒙特卡洛 Dropout 不确定性估计
        
        通过在推理时保持 Dropout 开启，进行多次前向传播，
        利用预测的方差来估计认知不确定性 (epistemic uncertainty)
        
        Args:
            x: 输入图像 [B, C, H, W]
            n_samples: 采样次数（越多越准确，但越慢）
            dropout_rate: 如果指定，临时修改 Dropout 比率
            
        Returns:
            dict: 包含均值、方差、熵等不确定性指标
        """
        x = x.to(self.device)
        
        # 保存原始 Dropout 状态
        original_dropout_states = []
        if dropout_rate is not None:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    original_dropout_states.append(module.p)
                    module.p = dropout_rate
        
        # 保持训练模式以启用 Dropout
        self.model.train()
        
        # 收集多次前向传播的结果
        probs_list = []
        logits_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)
                logits_list.append(logits)
        
        # 恢复评估模式
        self.model.eval()
        
        # 恢复原始 Dropout 比率
        if dropout_rate is not None:
            idx = 0
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = original_dropout_states[idx]
                    idx += 1
        
        # 堆叠结果 [n_samples, B, num_classes]
        probs_stack = torch.stack(probs_list, dim=0)
        logits_stack = torch.stack(logits_list, dim=0)
        
        # 计算统计量
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        var_probs = probs_stack.var(dim=0)
        
        # 计算预测熵
        eps = 1e-7
        entropy = - (mean_probs * torch.log(mean_probs + eps) + 
                    (1 - mean_probs) * torch.log(1 - mean_probs + eps))
        
        # 计算变异系数 (Coefficient of Variation)
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
        温度缩放校准
        
        学习一个温度参数 T，使预测概率更好地校准
        解决模型过度自信或过度不自信的问题
        
        Args:
            val_loader: 验证集数据加载器
            n_classes: 类别数
            max_iter: 最大迭代次数
            lr: 学习率
            
        Returns:
            float: 最优温度参数
        """
        # 初始化温度参数
        temperature = nn.Parameter(torch.ones(1).to(self.device))
        
        # 收集验证集 logits 和 labels
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
        
        # 优化温度参数
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
        print(f"温度缩放校准完成，最优温度: {optimal_temp:.4f}")
        
        return optimal_temp
    
    def get_uncertainty_summary(self, uncertainty_dict: Dict) -> Dict:
        """
        从不确定性结果中提取摘要统计
        
        Args:
            uncertainty_dict: mc_dropout 的返回结果
            
        Returns:
            dict: 不确定性摘要
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
        将不确定性分级（低/中/高）
        
        Args:
            uncertainty_dict: 不确定性估计结果
            low_threshold: 低不确定性阈值
            high_threshold: 高不确定性阈值
            
        Returns:
            list: 每个样本的不确定性等级
        """
        std = uncertainty_dict['std_probs']
        sample_uncertainty = std.mean(dim=1)  # 每个样本的平均不确定性
        
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
    任意不确定性 (Aleatoric Uncertainty) 估计器
    
    估计数据本身固有的不确定性（如噪声、模糊性）
    通过学习预测的方差来实现
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: 输出 logits 和 log_variance 的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
    
    def estimate(self, x: torch.Tensor) -> Dict:
        """
        估计任意不确定性
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            dict: 包含均值预测和不确定性
        """
        x = x.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            
            # 假设模型输出 (mean, log_var)
            if isinstance(output, tuple):
                mean, log_var = output
            else:
                # 如果模型只输出一个值，无法估计任意不确定性
                raise ValueError("模型需要输出均值和对数方差")
        
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
    计算期望校准误差 (ECE)
    
    衡量模型预测概率的校准程度
    ECE 越低，模型的概率估计越可靠
    
    Args:
        probs: 预测概率 [N, num_classes]
        labels: 真实标签 [N, num_classes]
        n_bins: 分箱数量
        
    Returns:
        float: ECE 值
    """
    # 对于多标签分类，展平所有类别
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()
    
    # 按概率分箱
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # 找到当前 bin 的样本
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs_flat > bin_lower) & (probs_flat <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            # 计算 bin 内的平均准确率和平均置信度
            avg_confidence = probs_flat[in_bin].mean()
            avg_accuracy = labels_flat[in_bin].float().mean()
            
            ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return ece.item()
