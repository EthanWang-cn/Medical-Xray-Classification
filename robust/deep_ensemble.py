# -*- coding: utf-8 -*-
"""
深度集成学习 (Deep Ensembles)
通过多个独立训练的模型进行投票，提升预测准确性和鲁棒性
同时提供基于模型分歧的不确定性估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class DeepEnsemble:
    """
    深度集成学习类
    
    实现原理：
    1. 训练多个具有不同初始化或不同数据增强的模型
    2. 推理时对所有模型的预测进行平均或加权
    3. 通过模型间的分歧程度估计不确定性
    
    优势：
    - 显著提升分类准确率
    - 提供可靠的不确定性估计
    - 对噪声和扰动更鲁棒
    """
    
    def __init__(self, model_fn, num_models=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_fn: 模型创建函数，调用后返回一个新的模型实例
            num_models: 集成模型数量
            device: 计算设备
        """
        self.model_fn = model_fn
        self.num_models = num_models
        self.device = device
        
        # 模型列表
        self.models: List[nn.Module] = []
        
        # 模型权重（用于加权集成）
        self.weights: Optional[List[float]] = None
    
    def create_models(self):
        """创建所有集成模型（不同初始化）"""
        self.models = []
        for i in range(self.num_models):
            model = self.model_fn()
            model = model.to(self.device)
            self.models.append(model)
        print(f"创建了 {self.num_models} 个集成模型")
    
    def load_models(self, checkpoint_paths: List[str]):
        """
        从检查点加载模型
        
        Args:
            checkpoint_paths: 模型权重文件路径列表
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
        print(f"成功加载 {self.num_models} 个集成模型")
    
    def set_weights(self, weights: List[float]):
        """
        设置各模型的权重（用于加权平均）
        
        Args:
            weights: 权重列表，长度应等于模型数量
        """
        if len(weights) != self.num_models:
            raise ValueError(f"权重数量 ({len(weights)}) 与模型数量 ({self.num_models}) 不匹配")
        
        # 归一化权重
        total = sum(weights)
        self.weights = [w / total for w in weights]
        print(f"设置模型权重: {self.weights}")
    
    def predict(self, x: torch.Tensor, return_all: bool = False) -> Dict:
        """
        集成预测
        
        Args:
            x: 输入图像 [B, C, H, W]
            return_all: 是否返回所有模型的单独预测
            
        Returns:
            dict: 包含集成预测结果和不确定性信息
        """
        if not self.models:
            raise RuntimeError("模型未加载！请先调用 create_models() 或 load_models()")
        
        x = x.to(self.device)
        
        # 收集所有模型的预测
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(x)
                probs = torch.sigmoid(logits)
                all_logits.append(logits)
                all_probs.append(probs)
        
        # 堆叠 [num_models, B, num_classes]
        probs_stack = torch.stack(all_probs, dim=0)
        logits_stack = torch.stack(all_logits, dim=0)
        
        # 计算集成预测
        if self.weights is not None:
            # 加权平均
            weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)
            mean_probs = (probs_stack * weights_tensor).sum(dim=0)
            mean_logits = (logits_stack * weights_tensor).sum(dim=0)
        else:
            # 简单平均
            mean_probs = probs_stack.mean(dim=0)
            mean_logits = logits_stack.mean(dim=0)
        
        # 计算不确定性指标
        std_probs = probs_stack.std(dim=0)  # 标准差（模型间分歧）
        
        # 计算预测熵
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
        获取预测的不确定性等级
        
        Args:
            x: 输入图像
            thresholds: 不确定性阈值，默认 [0.1, 0.2, 0.3]
            
        Returns:
            dict: 每个样本的不确定性等级
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3]
        
        result = self.predict(x)
        std_probs = result['std_probs']
        
        # 计算每个样本的平均不确定性
        sample_uncertainty = std_probs.mean(dim=1)  # [B]
        
        # 分级
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
        保存所有模型权重
        
        Args:
            save_dir: 保存目录
            prefix: 文件名前缀
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            path = save_path / f"{prefix}_{i}.pth"
            torch.save(model.state_dict(), path)
        
        print(f"已保存 {self.num_models} 个模型到 {save_dir}")
