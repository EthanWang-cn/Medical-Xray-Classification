# -*- coding: utf-8 -*-
"""
Demo 推理脚本
快速演示模型预测、不确定性估计和鲁棒性测试
无需训练，直接使用预训练模型或随机初始化模型进行演示
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.robust_module import RobustClassifier
from robust.tta import LightTTA
from robust.uncertainty import UncertaintyEstimator
from robust.adversarial import AdversarialTester
from utils.visualization import visualize_predictions


def load_demo_image(image_path=None, image_size=224):
    """
    加载演示图像
    
    Args:
        image_path: 图像路径，None 则生成随机图像
        image_size: 图像尺寸
        
    Returns:
        tuple: (图像张量, numpy 图像)
    """
    if image_path and os.path.exists(image_path):
        # 加载真实图像
        img = Image.open(image_path).convert('RGB')
        img = img.resize((image_size, image_size))
        img_np = np.array(img) / 255.0
    else:
        # 生成随机图像（用于演示）
        print("未提供图像，使用随机生成的演示图像...")
        img_np = np.random.rand(image_size, image_size, 3).astype(np.float32)
    
    # 转换为张量
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    
    # 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, img_np


def demo_standard_prediction(model, image_tensor, class_names, device):
    """演示标准预测"""
    print("\n" + "=" * 60)
    print("1. 标准预测")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.sigmoid(logits)
    
    probs_np = probs.cpu().numpy()[0]
    
    print("\n预测结果:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probs_np[i]:.4f}")
    
    # Top-3
    top3_idx = np.argsort(probs_np)[-3:][::-1]
    print("\nTop-3 预测:")
    for idx in top3_idx:
        print(f"  {class_names[idx]}: {probs_np[idx]:.4f}")
    
    return probs_np


def demo_mc_dropout_uncertainty(model, image_tensor, class_names, device, n_samples=30):
    """演示 MC Dropout 不确定性估计"""
    print("\n" + "=" * 60)
    print("2. MC Dropout 不确定性估计")
    print("=" * 60)
    
    estimator = UncertaintyEstimator(model, device)
    result = estimator.mc_dropout(image_tensor.to(device), n_samples=n_samples)
    
    mean_probs = result['mean_probs'].cpu().numpy()[0]
    std_probs = result['std_probs'].cpu().numpy()[0]
    
    print(f"\n采样次数: {n_samples}")
    print("\n预测均值与不确定性:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {mean_probs[i]:.4f} ± {std_probs[i]:.4f}")
    
    # 不确定性最高的类别
    most_uncertain_idx = np.argmax(std_probs)
    print(f"\n不确定性最高的类别: {class_names[most_uncertain_idx]} "
          f"(std = {std_probs[most_uncertain_idx]:.4f})")
    
    return result


def demo_tta(model, image_tensor, class_names, device):
    """演示测试时增强 (TTA)"""
    print("\n" + "=" * 60)
    print("3. 测试时增强 (TTA)")
    print("=" * 60)
    
    tta = LightTTA(model, device)
    result = tta.predict(image_tensor.to(device))
    
    tta_probs = result['probs'].cpu().numpy()[0]
    orig_probs = result['probs_original'].cpu().numpy()[0]
    
    print("\n标准预测 vs TTA 预测:")
    for i, name in enumerate(class_names):
        diff = tta_probs[i] - orig_probs[i]
        print(f"  {name}: {orig_probs[i]:.4f} → {tta_probs[i]:.4f} ({diff:+.4f})")
    
    # 平均变化
    mean_diff = np.mean(np.abs(tta_probs - orig_probs))
    print(f"\n平均概率变化: {mean_diff:.4f}")
    
    return result


def demo_adversarial_robustness(model, image_tensor, device):
    """演示对抗鲁棒性测试"""
    print("\n" + "=" * 60)
    print("4. 对抗鲁棒性测试")
    print("=" * 60)
    
    tester = AdversarialTester(model, device)
    
    # 生成伪标签（用于演示）
    dummy_label = torch.zeros(1, 14).to(device)
    dummy_label[0, 0] = 1  # 假设第一个类别为正
    
    # FGSM 攻击
    print("\nFGSM 攻击 (ε=0.03):")
    result = tester.evaluate_robustness(
        image_tensor.to(device), dummy_label,
        attack_type='fgsm', epsilon=0.03
    )
    print(f"  干净准确率: {result['accuracy_clean']:.4f}")
    print(f"  对抗准确率: {result['accuracy_adversarial']:.4f}")
    print(f"  准确率下降: {result['accuracy_drop']:.4f}")
    print(f"  预测变化率: {result['prediction_change_rate']:.4f}")
    
    # 高斯噪声
    print("\n高斯噪声 (σ=0.1):")
    result = tester.evaluate_robustness(
        image_tensor.to(device), dummy_label,
        attack_type='gaussian', std=0.1
    )
    print(f"  干净准确率: {result['accuracy_clean']:.4f}")
    print(f"  噪声准确率: {result['accuracy_adversarial']:.4f}")
    print(f"  准确率下降: {result['accuracy_drop']:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='医学影像鲁棒性分类 Demo')
    parser.add_argument('--image', type=str, default=None,
                        help='输入图像路径')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='模型架构')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径（可选）')
    parser.add_argument('--dataset', type=str, default='chestmnist',
                        help='数据集名称 (决定类别数和名称)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='图像尺寸')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='MC Dropout 采样次数')
    parser.add_argument('--output_dir', type=str, default='./results/demo',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 类别名称（ChestMNIST 的 14 种疾病）
    if args.dataset == 'chestmnist':
        class_names = [
            'atelectasis', 'cardiomegaly', 'effusion', 'infiltration',
            'mass', 'nodule', 'pneumonia', 'pneumothorax',
            'consolidation', 'edema', 'emphysema', 'fibrosis',
            'pleural_thickening', 'hernia'
        ]
        num_classes = 14
    elif args.dataset == 'dermamnist':
        class_names = [
            'actinic_keratoses', 'basal_cell_carcinoma',
            'benign_keratosis', 'dermatofibroma', 'melanoma',
            'melanocytic_nevi', 'vascular_lesions'
        ]
        num_classes = 7
    else:
        class_names = [f'class_{i}' for i in range(14)]
        num_classes = 14
    
    print(f"数据集: {args.dataset}")
    print(f"类别数: {num_classes}")
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = RobustClassifier(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=True,  # 使用 ImageNet 预训练权重
        dropout_rate=0.3,
        in_channels=3
    )
    model = model.to(device)
    
    # 加载检查点（如果提供）
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"加载检查点: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.model.load_state_dict(state_dict)
    else:
        print("使用 ImageNet 预训练权重（随机初始化分类头）")
    
    model.eval()
    
    # 加载图像
    print(f"\n加载图像...")
    image_tensor, image_np = load_demo_image(args.image, args.image_size)
    print(f"图像尺寸: {image_tensor.shape}")
    
    # 1. 标准预测
    probs = demo_standard_prediction(model, image_tensor, class_names, device)
    
    # 2. MC Dropout 不确定性估计
    unc_result = demo_mc_dropout_uncertainty(
        model, image_tensor, class_names, device, args.n_samples
    )
    
    # 3. TTA 演示
    tta_result = demo_tta(model, image_tensor, class_names, device)
    
    # 4. 对抗鲁棒性测试
    rob_result = demo_adversarial_robustness(model, image_tensor, device)
    
    # 保存演示图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 不确定性可视化
    std_map = unc_result['std_probs'].cpu().numpy()[0]
    axes[1].bar(range(num_classes), std_map)
    axes[1].set_title('Prediction Uncertainty (Std)')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Std')
    axes[1].set_xticks(range(num_classes))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=6)
    
    # 概率分布
    axes[2].bar(range(num_classes), probs)
    axes[2].set_title('Prediction Probabilities')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Probability')
    axes[2].set_xticks(range(num_classes))
    axes[2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=6)
    
    plt.tight_layout()
    save_path = output_dir / 'demo_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {save_path}")
    
    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)
    print("\n鲁棒性功能总结:")
    print("  ✓ 标准预测 (Standard Inference)")
    print("  ✓ MC Dropout 不确定性估计 (Uncertainty Estimation)")
    print("  ✓ 测试时增强 (Test-Time Augmentation)")
    print("  ✓ 对抗鲁棒性测试 (Adversarial Robustness)")
    print("\n更多功能请查看 train.py 和 evaluate.py")


if __name__ == '__main__':
    main()
