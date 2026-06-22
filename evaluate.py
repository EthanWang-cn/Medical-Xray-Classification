# -*- coding: utf-8 -*-
"""
Evaluation Script
Supports standard evaluation, robustness testing, uncertainty analysis, etc.
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import get_backbone
from robust.tta import LightTTA
from robust.uncertainty import UncertaintyEstimator
from robust.adversarial import AdversarialTester
from utils.metrics import calculate_metrics
from utils.visualization import (
    plot_roc_curves,
    plot_uncertainty_distribution,
    plot_robustness_comparison
)


def get_test_transform(image_size: int) -> A.Compose:
    """Get test-time transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class MedMNISTDataset(torch.utils.data.Dataset):
    """MedMNIST Dataset Wrapper"""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        label = torch.FloatTensor(label).squeeze()
        return img, label


def standard_evaluation(model, test_loader, device, class_names):
    """Standard evaluation"""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Standard Evaluation'):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = (all_probs > 0.5).astype(int)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs, class_names)
    
    return metrics, all_labels, all_probs


def tta_evaluation(model, test_loader, device, class_names):
    """TTA evaluation"""
    tta = LightTTA(model, device)
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='TTA Evaluation'):
            images = images.to(device)
            result = tta.predict(images)
            
            all_labels.append(labels.numpy())
            all_probs.append(result['probs'].cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = (all_probs > 0.5).astype(int)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs, class_names)
    
    return metrics, all_labels, all_probs


def uncertainty_analysis(model, test_loader, device, n_samples=30):
    """Uncertainty analysis"""
    estimator = UncertaintyEstimator(model, device)
    
    all_uncertainty = []
    
    # Only take a subset of samples for uncertainty analysis (to save time)
    max_samples = 500
    count = 0
    
    for images, labels in tqdm(test_loader, desc='Uncertainty Analysis'):
        if count >= max_samples:
            break
        
        images = images.to(device)
        result = estimator.mc_dropout(images, n_samples=n_samples)
        
        # Average uncertainty per sample
        sample_unc = result['std_probs'].mean(dim=1).cpu().numpy()
        all_uncertainty.append(sample_unc)
        
        count += len(images)
    
    all_uncertainty = np.concatenate(all_uncertainty, axis=0)
    
    return {
        'mean': float(np.mean(all_uncertainty)),
        'std': float(np.std(all_uncertainty)),
        'median': float(np.median(all_uncertainty)),
        'values': all_uncertainty
    }


def robustness_test(model, test_loader, device):
    """Robustness testing"""
    tester = AdversarialTester(model, device)
    
    # Take one batch of data for testing
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    results = {}
    
    # FGSM attack
    for eps in [0.01, 0.03, 0.05]:
        result = tester.evaluate_robustness(
            images, labels, attack_type='fgsm', epsilon=eps
        )
        results[f'FGSM (ε={eps})'] = {
            'accuracy_clean': result['accuracy_clean'],
            'accuracy_adversarial': result['accuracy_adversarial'],
            'accuracy_drop': result['accuracy_drop'],
            'prediction_change_rate': result['prediction_change_rate']
        }
    
    # PGD attack
    result = tester.evaluate_robustness(
        images, labels, attack_type='pgd', epsilon=0.03, steps=10
    )
    results['PGD (ε=0.03)'] = {
        'accuracy_clean': result['accuracy_clean'],
        'accuracy_adversarial': result['accuracy_adversarial'],
        'accuracy_drop': result['accuracy_drop'],
        'prediction_change_rate': result['prediction_change_rate']
    }
    
    # Gaussian noise
    for std in [0.05, 0.1]:
        result = tester.evaluate_robustness(
            images, labels, attack_type='gaussian', std=std
        )
        results[f'Gaussian Noise (σ={std})'] = {
            'accuracy_clean': result['accuracy_clean'],
            'accuracy_adversarial': result['accuracy_adversarial'],
            'accuracy_drop': result['accuracy_drop'],
            'prediction_change_rate': result['prediction_change_rate']
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Medical Image Robust Classification Evaluation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--tta', action='store_true',
                        help='Whether to use TTA')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Whether to perform uncertainty analysis')
    parser.add_argument('--robustness', action='store_true',
                        help='Whether to perform robustness testing')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.dataset:
        config['dataset']['name'] = args.dataset
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset information
    dataset_name = config['dataset']['name']
    info = INFO[dataset_name]
    n_classes = len(info['label'])
    class_names = list(info['label'].values())
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of classes: {n_classes}")
    
    # Load dataset
    image_size = config['dataset']['image_size']
    test_transform = get_test_transform(image_size)
    
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(
        split='test',
        transform=None,
        download=True,
        size=image_size,
        root=config['dataset']['data_dir']
    )
    test_dataset = MedMNISTDataset(test_dataset, test_transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Load model
    model = get_backbone(
        model_name=config['model']['backbone'],
        num_classes=n_classes,
        pretrained=False,
        dropout_rate=config['model']['dropout_rate'],
        in_channels=3
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {args.checkpoint}")
    
    # Results dictionary
    all_results = {}
    
    # 1. Standard evaluation
    print("\n" + "=" * 60)
    print("1. Standard Evaluation")
    print("=" * 60)
    
    std_metrics, std_labels, std_probs = standard_evaluation(
        model, test_loader, device, class_names
    )
    
    print(f"\nStandard Evaluation Results:")
    print(f"  Mean AUC: {std_metrics['overall']['mean_auc']:.4f}")
    print(f"  Mean F1:  {std_metrics['overall']['mean_f1']:.4f}")
    print(f"  Exact Match Ratio: {std_metrics['overall']['exact_match_ratio']:.4f}")
    print(f"  Hamming Loss: {std_metrics['overall']['hamming_loss']:.4f}")
    
    all_results['standard'] = {
        'mean_auc': std_metrics['overall']['mean_auc'],
        'mean_f1': std_metrics['overall']['mean_f1'],
        'exact_match_ratio': std_metrics['overall']['exact_match_ratio'],
        'hamming_loss': std_metrics['overall']['hamming_loss'],
        'per_class': {
            name: {
                'auc': m['auc'],
                'f1': m['f1'],
                'precision': m['precision'],
                'recall': m['recall']
            }
            for name, m in std_metrics['per_class'].items()
        }
    }
    
    # Plot ROC curves
    plot_roc_curves(
        std_labels, std_probs, class_names,
        save_path=str(figure_dir / 'roc_curves.png')
    )
    
    # 2. TTA evaluation
    if args.tta:
        print("\n" + "=" * 60)
        print("2. TTA Evaluation")
        print("=" * 60)
        
        tta_metrics, tta_labels, tta_probs = tta_evaluation(
            model, test_loader, device, class_names
        )
        
        print(f"\nTTA Evaluation Results:")
        print(f"  Mean AUC: {tta_metrics['overall']['mean_auc']:.4f}")
        print(f"  Mean F1:  {tta_metrics['overall']['mean_f1']:.4f}")
        
        # Comparison
        auc_improvement = tta_metrics['overall']['mean_auc'] - std_metrics['overall']['mean_auc']
        f1_improvement = tta_metrics['overall']['mean_f1'] - std_metrics['overall']['mean_f1']
        print(f"\nTTA Improvement:")
        print(f"  AUC Improvement: {auc_improvement:+.4f} ({auc_improvement*100:+.2f}%)")
        print(f"  F1 Improvement:  {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
        
        all_results['tta'] = {
            'mean_auc': tta_metrics['overall']['mean_auc'],
            'mean_f1': tta_metrics['overall']['mean_f1'],
            'auc_improvement': auc_improvement,
            'f1_improvement': f1_improvement
        }
    
    # 3. Uncertainty analysis
    if args.uncertainty:
        print("\n" + "=" * 60)
        print("3. Uncertainty Analysis (MC Dropout)")
        print("=" * 60)
        
        unc_result = uncertainty_analysis(
            model, test_loader, device,
            n_samples=config['robustness']['uncertainty']['n_samples']
        )
        
        print(f"\nUncertainty Statistics:")
        print(f"  Mean: {unc_result['mean']:.6f}")
        print(f"  Std:  {unc_result['std']:.6f}")
        print(f"  Median: {unc_result['median']:.6f}")
        
        all_results['uncertainty'] = {
            'method': 'MC Dropout',
            'n_samples': config['robustness']['uncertainty']['n_samples'],
            'mean': unc_result['mean'],
            'std': unc_result['std'],
            'median': unc_result['median']
        }
        
        # Plot uncertainty distribution
        plot_uncertainty_distribution(
            unc_result['values'],
            uncertainty_type='Prediction Std',
            save_path=str(figure_dir / 'uncertainty_distribution.png')
        )
    
    # 4. Robustness testing
    if args.robustness:
        print("\n" + "=" * 60)
        print("4. Robustness Testing")
        print("=" * 60)
        
        rob_results = robustness_test(model, test_loader, device)
        
        print("\nRobustness Test Results:")
        for name, res in rob_results.items():
            print(f"\n  {name}:")
            print(f"    Clean Accuracy: {res['accuracy_clean']:.4f}")
            print(f"    Adversarial Accuracy: {res['accuracy_adversarial']:.4f}")
            print(f"    Accuracy Drop: {res['accuracy_drop']:.4f}")
            print(f"    Prediction Change Rate: {res['prediction_change_rate']:.4f}")
        
        all_results['robustness'] = rob_results
        
        # Plot robustness comparison
        plot_robustness_comparison(
            rob_results,
            save_path=str(figure_dir / 'robustness_comparison.png')
        )
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    
    # Convert numpy types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    all_results_serializable = convert_to_serializable(all_results)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to: {results_path}")
    print(f"Visualization figures saved to: {figure_dir}")


if __name__ == '__main__':
    main()
