# -*- coding: utf-8 -*-
"""
鲁棒性分类器模块
集成多种鲁棒性技术，提供统一的推理接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbone import get_backbone


class RobustClassifier(nn.Module):
    """
    鲁棒性医学影像分类器
    
    集成多种鲁棒性增强技术：
    1. 深度集成学习 (Deep Ensembles)
    2. 蒙特卡洛 Dropout 不确定性估计
    3. 测试时增强 (Test-Time Augmentation, TTA)
    4. 对抗扰动鲁棒性测试
    
    提供统一的推理接口，支持多种鲁棒性模式切换
    """
    
    def __init__(self, model_name='resnet50', num_classes=14,
                 pretrained=True, dropout_rate=0.3, in_channels=3,
                 n_ensembles=5):
        """
        Args:
            model_name: 骨干网络名称
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout 比率
            in_channels: 输入通道数
            n_ensembles: 集成模型数量（用于深度集成）
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.n_ensembles = n_ensembles
        self.dropout_rate = dropout_rate
        
        # 创建单个模型（单模型模式）
        self.model = get_backbone(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            in_channels=in_channels
        )
        
        # 深度集成模型列表（用于 Deep Ensembles）
        self.ensemble_models = nn.ModuleList()
        
        # 标记是否已加载集成模型
        self._ensemble_loaded = False
    
    def forward(self, x):
        """
        标准前向传播（单模型）
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            logits: 分类 logits [B, num_classes]
        """
        return self.model(x)
    
    def predict_proba(self, x, mode='standard', **kwargs):
        """
        预测概率，支持多种鲁棒性模式
        
        Args:
            x: 输入图像 [B, C, H, W]
            mode: 推理模式
                - 'standard': 标准推理
                - 'mc_dropout': 蒙特卡洛 Dropout 不确定性估计
                - 'ensemble': 深度集成推理
                - 'tta': 测试时增强
            **kwargs: 各模式的额外参数
            
        Returns:
            dict: 包含预测概率和不确定性信息的字典
        """
        if mode == 'standard':
            return self._standard_predict(x)
        elif mode == 'mc_dropout':
            n_samples = kwargs.get('n_samples', 30)
            return self._mc_dropout_predict(x, n_samples)
        elif mode == 'ensemble':
            return self._ensemble_predict(x)
        else:
            raise ValueError(f"不支持的推理模式: {mode}")
    
    def _standard_predict(self, x):
        """标准推理"""
        self.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)  # 多标签分类用 sigmoid
        
        return {
            'probs': probs,
            'logits': logits,
            'uncertainty': None
        }
    
    def _mc_dropout_predict(self, x, n_samples=30):
        """
        蒙特卡洛 Dropout 不确定性估计
        
        通过多次前向传播（保持 Dropout 开启），
        估计预测的不确定性（认知不确定性）
        
        Args:
            x: 输入图像 [B, C, H, W]
            n_samples: 采样次数
            
        Returns:
            dict: 包含均值概率、方差、熵等不确定性指标
        """
        # 保持训练模式以启用 Dropout
        self.train()
        
        # 收集多次前向传播的结果
        probs_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)
        
        # 堆叠所有采样结果 [n_samples, B, num_classes]
        probs_stack = torch.stack(probs_list, dim=0)
        
        # 计算统计量
        mean_probs = probs_stack.mean(dim=0)       # 均值概率
        std_probs = probs_stack.std(dim=0)         # 标准差
        var_probs = probs_stack.var(dim=0)         # 方差
        
        # 计算预测熵（不确定性的另一种度量）
        # 对于多标签分类，每个类别独立计算
        eps = 1e-7
        entropy = - (mean_probs * torch.log(mean_probs + eps) + 
                    (1 - mean_probs) * torch.log(1 - mean_probs + eps))
        
        # 切换回评估模式
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
        加载深度集成模型
        
        Args:
            model_paths: 模型权重路径列表
        """
        if len(model_paths) != self.n_ensembles:
            print(f"警告: 预期 {self.n_ensembles} 个模型，实际加载 {len(model_paths)} 个")
        
        self.ensemble_models = nn.ModuleList()
        
        for path in model_paths:
            # 创建新模型实例
            model = get_backbone(
                model_name=self.model.__class__.__name__.replace('Classifier', '').lower(),
                num_classes=self.num_classes,
                pretrained=False,
                dropout_rate=self.dropout_rate
            )
            # 加载权重
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            self.ensemble_models.append(model)
        
        self._ensemble_loaded = True
        print(f"成功加载 {len(self.ensemble_models)} 个集成模型")
    
    def _ensemble_predict(self, x):
        """
        深度集成推理
        
        通过多个独立训练的模型进行投票，
        同时提供不确定性估计
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            dict: 集成预测结果及不确定性
        """
        if not self._ensemble_loaded:
            raise RuntimeError(
                "集成模型未加载！请先调用 load_ensemble() 加载模型权重"
            )
        
        probs_list = []
        
        with torch.no_grad():
            for model in self.ensemble_models:
                model.eval()
                logits = model(x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)
        
        # 堆叠结果 [n_ensembles, B, num_classes]
        probs_stack = torch.stack(probs_list, dim=0)
        
        # 计算统计量
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        
        # 计算预测多样性（模型间分歧）
        # 使用熵来度量集成的不确定性
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
        从预测结果中提取不确定性摘要
        
        Args:
            result_dict: predict_proba 的返回结果
            
        Returns:
            dict: 不确定性摘要统计
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
