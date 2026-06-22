# -*- coding: utf-8 -*-
"""
评估指标模块
实现医学影像分类常用的评估指标
包括 AUC、F1、准确率、召回率、精确率等
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score
)
from typing import Dict, List, Optional, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                      y_prob: np.ndarray, class_names: Optional[List[str]] = None) -> Dict:
    """
    计算多标签分类的综合评估指标
    
    Args:
        y_true: 真实标签 [N, num_classes]
        y_pred: 预测标签 [N, num_classes]
        y_prob: 预测概率 [N, num_classes]
        class_names: 类别名称列表
        
    Returns:
        dict: 包含各类指标的字典
    """
    num_classes = y_true.shape[1]
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    metrics = {
        'per_class': {},
        'overall': {}
    }
    
    # 逐类指标
    for i, name in enumerate(class_names):
        class_metrics = {}
        
        # AUC
        try:
            class_metrics['auc'] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            class_metrics['auc'] = 0.0
        
        # F1
        class_metrics['f1'] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        
        # 精确率
        class_metrics['precision'] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        
        # 召回率
        class_metrics['recall'] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        
        # 准确率
        class_metrics['accuracy'] = accuracy_score(y_true[:, i], y_pred[:, i])
        
        # AP (Average Precision)
        try:
            class_metrics['ap'] = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            class_metrics['ap'] = 0.0
        
        metrics['per_class'][name] = class_metrics
    
    # 整体指标
    metrics['overall'] = {
        'mean_auc': np.mean([m['auc'] for m in metrics['per_class'].values()]),
        'mean_f1': np.mean([m['f1'] for m in metrics['per_class'].values()]),
        'mean_precision': np.mean([m['precision'] for m in metrics['per_class'].values()]),
        'mean_recall': np.mean([m['recall'] for m in metrics['per_class'].values()]),
        'mean_ap': np.mean([m['ap'] for m in metrics['per_class'].values()]),
        'exact_match_ratio': _exact_match_ratio(y_true, y_pred),
        'hamming_loss': _hamming_loss(y_true, y_pred)
    }
    
    return metrics


def calculate_auc(y_true: np.ndarray, y_prob: np.ndarray, 
                  class_names: Optional[List[str]] = None) -> Dict:
    """
    计算 AUC 指标
    
    Args:
        y_true: 真实标签 [N, num_classes]
        y_prob: 预测概率 [N, num_classes]
        class_names: 类别名称列表
        
    Returns:
        dict: AUC 指标
    """
    num_classes = y_true.shape[1]
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    aucs = {}
    for i, name in enumerate(class_names):
        try:
            aucs[name] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            aucs[name] = 0.0
    
    aucs['mean'] = np.mean(list(aucs.values()))
    
    return aucs


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: Optional[List[str]] = None,
                 average: str = 'macro') -> Dict:
    """
    计算 F1 分数
    
    Args:
        y_true: 真实标签 [N, num_classes]
        y_pred: 预测标签 [N, num_classes]
        class_names: 类别名称列表
        average: 平均方式 ('macro', 'micro', 'weighted')
        
    Returns:
        dict: F1 指标
    """
    num_classes = y_true.shape[1]
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    f1s = {}
    for i, name in enumerate(class_names):
        f1s[name] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    
    f1s[f'{average}_avg'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return f1s


def _exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算精确匹配率（所有标签都预测正确的样本比例）
    
    Args:
        y_true: 真实标签 [N, num_classes]
        y_pred: 预测标签 [N, num_classes]
        
    Returns:
        float: 精确匹配率
    """
    exact_matches = np.all(y_true == y_pred, axis=1)
    return np.mean(exact_matches)


def _hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算汉明损失（错误预测的标签比例）
    
    越低越好
    
    Args:
        y_true: 真实标签 [N, num_classes]
        y_pred: 预测标签 [N, num_classes]
        
    Returns:
        float: 汉明损失
    """
    return np.mean(y_true != y_pred)


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                         metric: str = 'f1') -> np.ndarray:
    """
    寻找最优分类阈值
    
    Args:
        y_true: 真实标签 [N, num_classes]
        y_prob: 预测概率 [N, num_classes]
        metric: 优化指标 ('f1', 'f0.5', 'f2')
        
    Returns:
        np.ndarray: 每个类别的最优阈值 [num_classes]
    """
    num_classes = y_true.shape[1]
    best_thresholds = np.zeros(num_classes)
    
    thresholds = np.arange(0.0, 1.0, 0.01)
    
    for i in range(num_classes):
        best_score = -1
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_prob[:, i] >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true[:, i], y_pred, zero_division=0)
            elif metric == 'f0.5':
                score = f1_score(y_true[:, i], y_pred, beta=0.5, zero_division=0)
            elif metric == 'f2':
                score = f1_score(y_true[:, i], y_pred, beta=2, zero_division=0)
            else:
                score = f1_score(y_true[:, i], y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        best_thresholds[i] = best_thresh
    
    return best_thresholds


def robustness_metrics(clean_acc: float, adv_acc: float) -> Dict:
    """
    计算鲁棒性相关指标
    
    Args:
        clean_acc: 干净样本上的准确率
        adv_acc: 对抗样本上的准确率
        
    Returns:
        dict: 鲁棒性指标
    """
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'accuracy_drop': clean_acc - adv_acc,
        'relative_drop': (clean_acc - adv_acc) / clean_acc if clean_acc > 0 else 0,
        'robustness_ratio': adv_acc / clean_acc if clean_acc > 0 else 0
    }
