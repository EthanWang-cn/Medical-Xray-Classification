# -*- coding: utf-8 -*-
"""
骨干网络实现
支持 ResNet、DenseNet、EfficientNet 等多种经典架构
基于 timm 库实现，支持预训练权重加载
"""

import torch
import torch.nn as nn
import timm


class BaseClassifier(nn.Module):
    """分类器基类，封装通用的分类头结构"""
    
    def __init__(self, num_classes, in_features, dropout_rate=0.3):
        """
        Args:
            num_classes: 分类类别数
            in_features: 输入特征维度
            dropout_rate: Dropout 比率，用于正则化和不确定性估计
        """
        super().__init__()
        self.num_classes = num_classes
        
        # 分类头：全局平均池化 + Dropout + 全连接层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),             # 展平
            nn.Dropout(p=dropout_rate),  # Dropout 层（用于 MC Dropout 不确定性估计）
            nn.Linear(in_features, num_classes)  # 分类全连接层
        )
    
    def forward(self, x):
        """前向传播"""
        return self.classifier(x)


class ResNetClassifier(nn.Module):
    """
    ResNet 系列分类器
    支持 resnet18, resnet34, resnet50, resnet101 等
    """
    
    def __init__(self, model_name='resnet50', num_classes=14, 
                 pretrained=True, dropout_rate=0.3, in_channels=3):
        """
        Args:
            model_name: 模型名称，如 'resnet50', 'resnet18'
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout 比率
            in_channels: 输入通道数（医学影像可能是单通道）
        """
        super().__init__()
        
        # 使用 timm 创建骨干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 不使用默认分类头
            in_chans=in_channels
        )
        
        # 获取特征维度
        if hasattr(self.backbone, 'num_features'):
            in_features = self.backbone.num_features
        else:
            in_features = self.backbone.fc.in_features
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [B, C, H, W]
            
        Returns:
            logits: 分类 logits [B, num_classes]
        """
        # 提取特征
        features = self.backbone(x)
        # 分类
        logits = self.classifier(features)
        return logits
    
    def forward_features(self, x):
        """提取特征向量（用于可视化或迁移学习）"""
        return self.backbone(x)


class DenseNetClassifier(nn.Module):
    """
    DenseNet 系列分类器
    医学影像领域经典架构，CheXNet 即基于 DenseNet121
    """
    
    def __init__(self, model_name='densenet121', num_classes=14,
                 pretrained=True, dropout_rate=0.3, in_channels=3):
        """
        Args:
            model_name: 模型名称，如 'densenet121', 'densenet169'
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout 比率
            in_channels: 输入通道数
        """
        super().__init__()
        
        # 使用 timm 创建骨干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels
        )
        
        # 获取特征维度
        in_features = self.backbone.num_features
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet 系列分类器
    高效轻量，适合移动端部署
    """
    
    def __init__(self, model_name='efficientnet_b0', num_classes=14,
                 pretrained=True, dropout_rate=0.3, in_channels=3):
        """
        Args:
            model_name: 模型名称，如 'efficientnet_b0', 'efficientnet_b3'
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout 比率
            in_channels: 输入通道数
        """
        super().__init__()
        
        # 使用 timm 创建骨干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels
        )
        
        # 获取特征维度
        in_features = self.backbone.num_features
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_backbone(model_name='resnet50', num_classes=14, 
                 pretrained=True, dropout_rate=0.3, in_channels=3):
    """
    工厂函数：根据模型名称创建对应的分类器
    
    Args:
        model_name: 模型名称
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        dropout_rate: Dropout 比率
        in_channels: 输入通道数
        
    Returns:
        model: 分类器模型
        
    Raises:
        ValueError: 当模型名称不支持时抛出异常
    """
    model_name_lower = model_name.lower()
    
    if 'resnet' in model_name_lower:
        return ResNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            in_channels=in_channels
        )
    elif 'densenet' in model_name_lower:
        return DenseNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            in_channels=in_channels
        )
    elif 'efficientnet' in model_name_lower:
        return EfficientNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            in_channels=in_channels
        )
    else:
        # 尝试用 timm 直接创建
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
                drop_rate=dropout_rate
            )
            return model
        except Exception as e:
            raise ValueError(
                f"不支持的模型名称: {model_name}\n"
                f"支持的模型系列: ResNet, DenseNet, EfficientNet\n"
                f"或任意 timm 支持的模型名称"
            ) from e
