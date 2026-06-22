# -*- coding: utf-8 -*-
"""
可视化工具模块
包含 ROC 曲线、混淆矩阵、预测结果可视化等功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Dict, List, Optional, Tuple


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                    class_names: Optional[List[str]] = None,
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    绘制多标签分类的 ROC 曲线
    
    Args:
        y_true: 真实标签 [N, num_classes]
        y_prob: 预测概率 [N, num_classes]
        class_names: 类别名称列表
        save_path: 保存路径，None 则不保存
        figsize: 图像大小
        
    Returns:
        matplotlib Figure 对象
    """
    num_classes = y_true.shape[1]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制每个类别的 ROC 曲线
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # 绘制随机猜测线
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right', fontsize='small')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC 曲线已保存到: {save_path}")
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          normalize: bool = True,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    绘制混淆矩阵（针对单分类问题或单个类别）
    
    Args:
        y_true: 真实标签 [N] 或 [N, 1]
        y_pred: 预测标签 [N] 或 [N, 1]
        class_names: 类别名称
        normalize: 是否归一化
        save_path: 保存路径
        figsize: 图像大小
        
    Returns:
        matplotlib Figure 对象
    """
    # 确保是一维数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    return fig


def visualize_predictions(images: np.ndarray, y_true: np.ndarray, 
                          y_prob: np.ndarray, class_names: List[str],
                          n_samples: int = 8, cols: int = 4,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    可视化预测结果
    
    Args:
        images: 图像数组 [N, H, W, C] 或 [N, C, H, W]
        y_true: 真实标签 [N, num_classes]
        y_prob: 预测概率 [N, num_classes]
        class_names: 类别名称列表
        n_samples: 显示样本数
        cols: 列数
        save_path: 保存路径
        figsize: 图像大小
        
    Returns:
        matplotlib Figure 对象
    """
    n_samples = min(n_samples, len(images))
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    # 随机选择样本
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    for idx, ax in zip(indices, axes):
        img = images[idx]
        
        # 处理通道顺序
        if img.shape[0] in [1, 3]:  # CHW 格式
            img = np.transpose(img, (1, 2, 0))
        
        # 处理单通道
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        # 获取预测结果
        true_labels = y_true[idx]
        pred_probs = y_prob[idx]
        
        # 找出概率最高的几个类别
        top_k = min(3, len(class_names))
        top_indices = np.argsort(pred_probs)[-top_k:][::-1]
        
        # 构建标题
        title_parts = []
        for i in top_indices:
            prob = pred_probs[i]
            true = true_labels[i]
            marker = "✓" if true == 1 else ""
            title_parts.append(f"{class_names[i]}: {prob:.2f}{marker}")
        
        title = "\n".join(title_parts)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # 隐藏多余的子图
    for ax in axes[n_samples:]:
        ax.axis('off')
    
    plt.suptitle('Prediction Visualization (Top-3 probabilities)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测可视化已保存到: {save_path}")
    
    return fig


def plot_uncertainty_distribution(uncertainty_values: np.ndarray,
                                  uncertainty_type: str = 'std',
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制不确定性分布直方图
    
    Args:
        uncertainty_values: 不确定性值数组
        uncertainty_type: 不确定性类型名称
        save_path: 保存路径
        figsize: 图像大小
        
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(uncertainty_values.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(uncertainty_values), color='red', linestyle='--', 
               label=f'Mean: {np.mean(uncertainty_values):.4f}')
    ax.axvline(np.median(uncertainty_values), color='orange', linestyle='--',
               label=f'Median: {np.median(uncertainty_values):.4f}')
    
    ax.set_xlabel(f'{uncertainty_type} Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {uncertainty_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"不确定性分布图已保存到: {save_path}")
    
    return fig


def plot_robustness_comparison(robustness_results: Dict,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    绘制鲁棒性对比图
    
    Args:
        robustness_results: 鲁棒性测试结果字典
        save_path: 保存路径
        figsize: 图像大小
        
    Returns:
        matplotlib Figure 对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    attack_names = list(robustness_results.keys())
    clean_accs = [robustness_results[name]['accuracy_clean'] for name in attack_names]
    adv_accs = [robustness_results[name]['accuracy_adversarial'] for name in attack_names]
    drops = [robustness_results[name]['accuracy_drop'] for name in attack_names]
    
    x = np.arange(len(attack_names))
    width = 0.35
    
    # 左图：准确率对比
    ax1.bar(x - width/2, clean_accs, width, label='Clean', alpha=0.8)
    ax1.bar(x + width/2, adv_accs, width, label='Adversarial', alpha=0.8)
    ax1.set_xlabel('Attack Type')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Clean vs Adversarial Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(attack_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：准确率下降
    ax2.bar(x, drops, alpha=0.8, color='orange')
    ax2.set_xlabel('Attack Type')
    ax2.set_ylabel('Accuracy Drop')
    ax2.set_title('Accuracy Drop under Attacks')
    ax2.set_xticks(x)
    ax2.set_xticklabels(attack_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"鲁棒性对比图已保存到: {save_path}")
    
    return fig
