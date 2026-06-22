# -*- coding: utf-8 -*-
"""
训练脚本
支持多种医学影像数据集和模型架构
集成数据增强、学习率调度、早停等训练技巧
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import medmnist
from medmnist import INFO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import get_backbone
from utils.metrics import calculate_metrics


def set_seed(seed: int):
    """设置随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(image_size: int, is_train: bool = True) -> A.Compose:
    """
    获取数据增强变换
    
    Args:
        image_size: 图像尺寸
        is_train: 是否为训练模式
        
    Returns:
        albumentations 变换组合
    """
    if is_train:
        # 训练时的数据增强
        transform = A.Compose([
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # 验证/测试时的变换
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return transform


class MedMNISTDataset(torch.utils.data.Dataset):
    """
    MedMNIST 数据集包装类
    适配 albumentations 数据增强
    """
    
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset: MedMNIST 数据集对象
            transform: albumentations 变换
        """
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # 转换为 numpy 数组 (H, W, C)
        img = np.array(img)
        
        # 确保是 3 通道（如果是单通道则复制到 3 通道）
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        # 标签转换为 float（多标签分类）
        label = torch.FloatTensor(label).squeeze()
        
        return img, label


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        
    Returns:
        float: 平均损失
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        n_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / n_batches
    return avg_loss


def validate(model, dataloader, criterion, device, class_names=None):
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        class_names: 类别名称
        
    Returns:
        dict: 验证指标
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Validate]')
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            logits = model(images)
            loss = criterion(logits, labels)
            
            # 统计
            total_loss += loss.item()
            n_batches += 1
            
            # 收集预测
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算指标
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = (all_probs > 0.5).astype(int)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs, class_names)
    metrics['loss'] = total_loss / n_batches
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='医学影像鲁棒性分类训练')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集名称 (覆盖配置文件)')
    parser.add_argument('--model', type=str, default=None,
                        help='模型名称 (覆盖配置文件)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (覆盖配置文件)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小 (覆盖配置文件)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (覆盖配置文件)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子 (覆盖配置文件)')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.model:
        config['model']['backbone'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.seed:
        config['seed'] = args.seed
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建目录
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据集信息
    dataset_name = config['dataset']['name']
    info = INFO[dataset_name]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    task = info['task']
    class_names = list(info['label'].values())
    
    print(f"数据集: {dataset_name}")
    print(f"任务类型: {task}")
    print(f"类别数: {n_classes}")
    print(f"类别名称: {class_names}")
    
    # 数据变换
    image_size = config['dataset']['image_size']
    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)
    
    # 加载数据集
    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(
        split='train',
        transform=None,
        download=config['dataset']['download'],
        size=image_size,
        root=config['dataset']['data_dir']
    )
    val_dataset = DataClass(
        split='val',
        transform=None,
        download=config['dataset']['download'],
        size=image_size,
        root=config['dataset']['data_dir']
    )
    
    # 包装数据集
    train_dataset = MedMNISTDataset(train_dataset, train_transform)
    val_dataset = MedMNISTDataset(val_dataset, val_transform)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = get_backbone(
        model_name=config['model']['backbone'],
        num_classes=n_classes,
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate'],
        in_channels=3  # 统一使用 3 通道
    )
    model = model.to(device)
    
    # 损失函数（多标签分类用 BCEWithLogitsLoss）
    criterion = nn.BCEWithLogitsLoss()
    
    # 优化器
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 学习率调度器
    scheduler_type = config['training']['scheduler'].lower()
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-6
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = None
    
    # 训练循环
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping_patience']
    
    print("\n开始训练...")
    print("=" * 60)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, criterion, device, class_names
        )
        
        val_loss = val_metrics['loss']
        val_auc = val_metrics['overall']['mean_auc']
        val_f1 = val_metrics['overall']['mean_f1']
        
        # 更新学习率
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印结果
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  验证 AUC:  {val_auc:.4f}")
        print(f"  验证 F1:   {val_f1:.4f}")
        print(f"  学习率:    {current_lr:.6f}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型
            save_path = checkpoint_dir / f'best_model_{dataset_name}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  ★ 最佳模型已保存 (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= early_stopping_patience:
            print(f"\n早停触发！{early_stopping_patience} 轮没有提升")
            break
        
        print("-" * 60)
    
    print("\n训练完成！")
    print(f"最佳 epoch: {best_epoch}")
    print(f"最佳验证 AUC: {best_auc:.4f}")
    print(f"模型保存在: {checkpoint_dir / f'best_model_{dataset_name}.pth'}")


if __name__ == '__main__':
    main()
