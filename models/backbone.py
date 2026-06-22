# -*- coding: utf-8 -*-
"""
Backbone Network Implementations
Supports ResNet, DenseNet, EfficientNet and other classic architectures
Implemented using the timm library with pretrained weight support
"""
import torch
import torch.nn as nn
import timm


class BaseClassifier(nn.Module):
    """Base classifier class, encapsulating common classification head structure"""
    
    def __init__(self, num_classes, in_features, dropout_rate=0.3):
        """
        Args:
            num_classes: Number of classification classes
            in_features: Input feature dimension
            dropout_rate: Dropout rate for regularization and uncertainty estimation
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Classification head: Global Average Pooling + Dropout + Fully Connected
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),             # Flatten
            nn.Dropout(p=dropout_rate),  # Dropout layer (for MC Dropout uncertainty estimation)
            nn.Linear(in_features, num_classes)  # Classification fully connected layer
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.classifier(x)


class ResNetClassifier(nn.Module):
    """
    ResNet Series Classifier
    Supports resnet18, resnet34, resnet50, resnet101, etc.
    """
    
    def __init__(self, model_name='resnet50', num_classes=14, 
                 pretrained=True, dropout_rate=0.3, in_channels=3):
        """
        Args:
            model_name: Model name, e.g. 'resnet50', 'resnet18'
            num_classes: Number of classification classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
            in_channels: Number of input channels (medical images may be single-channel)
        """
        super().__init__()
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Do not use default classification head
            in_chans=in_channels
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            in_features = self.backbone.num_features
        else:
            in_features = self.backbone.fc.in_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)
        # Classify
        logits = self.classifier(features)
        return logits
    
    def forward_features(self, x):
        """Extract feature vectors (for visualization or transfer learning)"""
        return self.backbone(x)


class DenseNetClassifier(nn.Module):
    """
    DenseNet Series Classifier
    Classic architecture in medical imaging, CheXNet is based on DenseNet121
    """
    
    def __init__(self, model_name='densenet121', num_classes=14,
                 pretrained=True, dropout_rate=0.3, in_channels=3):
        """
        Args:
            model_name: Model name, e.g. 'densenet121', 'densenet169'
            num_classes: Number of classification classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
            in_channels: Number of input channels
        """
        super().__init__()
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels
        )
        
        # Get feature dimension
        in_features = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet Series Classifier
    Efficient and lightweight, suitable for mobile deployment
    """
    
    def __init__(self, model_name='efficientnet_b0', num_classes=14,
                 pretrained=True, dropout_rate=0.3, in_channels=3):
        """
        Args:
            model_name: Model name, e.g. 'efficientnet_b0', 'efficientnet_b3'
            num_classes: Number of classification classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
            in_channels: Number of input channels
        """
        super().__init__()
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels
        )
        
        # Get feature dimension
        in_features = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_backbone(model_name='resnet50', num_classes=14, 
                 pretrained=True, dropout_rate=0.3, in_channels=3):
    """
    Factory function: Create corresponding classifier based on model name
    
    Args:
        model_name: Model name
        num_classes: Number of classification classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate
        in_channels: Number of input channels
        
    Returns:
        model: Classifier model
        
    Raises:
        ValueError: Throws exception when model name is not supported
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
        # Try to create directly with timm
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
                f"Unsupported model name: {model_name}\n"
                f"Supported model families: ResNet, DenseNet, EfficientNet\n"
                f"Or any timm-supported model name"
            ) from e
