# -*- coding: utf-8 -*-
"""
测试时增强 (Test-Time Augmentation, TTA)
在推理阶段对输入图像进行多种变换，然后集成预测结果
提升模型对图像变化的鲁棒性
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
    测试时增强类
    
    实现原理：
    1. 对同一张测试图像应用多种不同的增强变换
    2. 对每个变换后的图像分别进行预测
    3. 将所有预测结果进行平均或投票
    
    优势：
    - 提升预测准确性（通常 0.5-2%）
    - 增强对图像平移、旋转、缩放的鲁棒性
    - 无需重新训练模型
    """
    
    def __init__(self, model: nn.Module, image_size: int = 224, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: 分类模型
            image_size: 图像尺寸
            device: 计算设备
        """
        self.model = model
        self.image_size = image_size
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        
        # 定义 TTA 变换管线
        self.transforms = self._build_tta_transforms()
    
    def _build_tta_transforms(self) -> List[Callable]:
        """
        构建 TTA 变换列表
        
        包含多种常见的测试时增强策略：
        - 原图
        - 水平翻转
        - 垂直翻转
        - 多尺度裁剪
        - 颜色微调
        """
        transforms_list = []
        
        # 1. 原始图像（中心裁剪）
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.CenterCrop(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 2. 水平翻转
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.CenterCrop(self.image_size, self.image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 3. 垂直翻转
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.CenterCrop(self.image_size, self.image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 4. 左上裁剪
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.Crop(0, 0, self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 5. 右下裁剪
        transforms_list.append(
            A.Compose([
                A.Resize(self.image_size + 32, self.image_size + 32),
                A.Crop(32, 32, self.image_size + 32, self.image_size + 32),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        )
        
        # 6. 轻微旋转 + 中心裁剪
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
        添加自定义 TTA 变换
        
        Args:
            transform: albumentations 变换组合
        """
        self.transforms.append(transform)
        print(f"添加自定义变换，当前共 {len(self.transforms)} 种变换")
    
    def predict(self, images: List[np.ndarray], weights: Optional[List[float]] = None) -> Dict:
        """
        TTA 预测
        
        Args:
            images: 输入图像列表（numpy 数组，HWC 格式，RGB）
            weights: 各变换的权重，None 表示等权重
            
        Returns:
            dict: TTA 集成预测结果
        """
        if weights is not None and len(weights) != len(self.transforms):
            raise ValueError(f"权重数量 ({len(weights)}) 与变换数量 ({len(self.transforms)}) 不匹配")
        
        all_probs = []
        
        with torch.no_grad():
            for i, transform in enumerate(self.transforms):
                # 应用变换
                augmented_batch = []
                for img in images:
                    augmented = transform(image=img)['image']
                    augmented_batch.append(augmented)
                
                # 堆叠成 batch
                batch = torch.stack(augmented_batch).to(self.device)
                
                # 预测
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
        
        # 堆叠 [n_transforms, B, num_classes]
        probs_stack = torch.stack(all_probs, dim=0)
        
        # 计算集成预测
        if weights is not None:
            weights_tensor = torch.tensor(weights, device=self.device).view(-1, 1, 1)
            mean_probs = (probs_stack * weights_tensor).sum(dim=0)
        else:
            mean_probs = probs_stack.mean(dim=0)
        
        # 计算不确定性（变换间的方差）
        std_probs = probs_stack.std(dim=0)
        
        return {
            'probs': mean_probs,
            'std_probs': std_probs,
            'num_transforms': len(self.transforms),
            'all_probs': probs_stack
        }
    
    def predict_single(self, image: np.ndarray, weights: Optional[List[float]] = None) -> Dict:
        """
        单张图像的 TTA 预测
        
        Args:
            image: 输入图像（numpy 数组，HWC 格式，RGB）
            weights: 各变换的权重
            
        Returns:
            dict: TTA 预测结果
        """
        return self.predict([image], weights)


class LightTTA:
    """
    轻量级 TTA（仅水平翻转）
    适合对速度要求较高的场景
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
    
    def predict(self, x: torch.Tensor) -> Dict:
        """
        轻量级 TTA 预测（仅水平翻转）
        
        Args:
            x: 输入图像张量 [B, C, H, W]
            
        Returns:
            dict: TTA 预测结果
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # 原图预测
            logits1 = self.model(x)
            probs1 = torch.sigmoid(logits1)
            
            # 水平翻转预测
            x_flip = torch.flip(x, dims=[3])  # 水平翻转
            logits2 = self.model(x_flip)
            probs2 = torch.sigmoid(logits2)
        
        # 平均
        mean_probs = (probs1 + probs2) / 2.0
        
        return {
            'probs': mean_probs,
            'probs_original': probs1,
            'probs_flipped': probs2,
            'num_transforms': 2
        }
