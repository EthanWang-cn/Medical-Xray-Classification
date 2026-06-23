# Med-Robust-CV: Robust Medical Image Classification with Uncertainty Estimation

> A PyTorch-based framework for robust medical image classification, featuring deep ensembles, uncertainty estimation, test-time augmentation, and adversarial robustness evaluation. Built on the MedMNIST benchmark.

## 📋 Overview

**Med-Robust-CV** is a comprehensive deep learning framework designed for robust medical image classification. Built upon the [MedMNIST benchmark](https://medmnist.com/), this project addresses a critical challenge in clinical AI: **model reliability under real-world conditions**. While deep learning models achieve impressive performance on standard benchmarks, their deployment in clinical settings requires more than just high accuracy — it requires **robustness**, **calibrated uncertainty estimates**, and **transparency**.

### Why This Matters

In medical imaging, model failures can have serious consequences. A model that is "confidently wrong" is far more dangerous than one that acknowledges its uncertainty. This project focuses on four pillars of reliable medical AI:

1. **Robustness** — Performance stability under distribution shifts, noise, and adversarial perturbations
2. **Uncertainty Quantification** — Knowing when the model doesn't know (critical for clinical decision support)
3. **Generalization** — Consistent performance across diverse patient populations and imaging protocols
4. **Transparency** — Interpretable predictions with confidence scores

### Key Features

- 🩻 **Multi-dataset support**: ChestMNIST (14 pathologies), DermaMNIST (7 skin lesions), PneumoniaMNIST (binary classification), and more from the MedMNIST benchmark
- 🏗️ **Multiple backbones**: ResNet, DenseNet, EfficientNet families via `timm`
- 🛡️ **Robustness modules**:
  - ✅ **Deep Ensembles** — Multi-model voting for improved accuracy and uncertainty
  - ✅ **MC Dropout Uncertainty** — Bayesian approximation for epistemic uncertainty estimation
  - ✅ **Test-Time Augmentation (TTA)** — Inference-time augmentation for stability
  - ✅ **Adversarial Robustness Testing** — FGSM, PGD, Gaussian noise, salt-and-pepper evaluation
- 📊 **Comprehensive evaluation**: AUC, F1, precision, recall, ECE (Expected Calibration Error)
- 🎨 **Visualization tools**: ROC curves, uncertainty distributions, robustness comparisons
- 📝 **Well-documented code**: Extensive comments for educational and research use

---

## 🏗️ Project Structure

```
med-robust-cv/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore configuration
├── configs/
│   └── config.yaml        # Configuration (dataset, model, training)
├── data/
│   └── README.md          # Dataset documentation
├── models/
│   ├── __init__.py
│   ├── backbone.py        # Backbone networks (ResNet/DenseNet/EfficientNet)
│   └── robust_module.py   # Robust classifier wrapper
├── robust/
│   ├── __init__.py
│   ├── deep_ensemble.py   # Deep Ensembles implementation
│   ├── tta.py             # Test-Time Augmentation
│   ├── uncertainty.py     # Uncertainty estimation (MC Dropout, temperature scaling)
│   └── adversarial.py     # Adversarial attacks & robustness testing
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # Evaluation metrics (AUC, F1, ECE, etc.)
│   └── visualization.py   # Visualization utilities
├── train.py               # Training script
├── evaluate.py            # Evaluation script
└── demo.py                # Quick demo script
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or initialize the repository
git init  # Initialize git in the extracted directory

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

Experience all robustness features without training:

```bash
# Basic demo with random image
python demo.py

# With your own image
python demo.py --image path/to/your/image.png

# Specify dataset and model
python demo.py --dataset chestmnist --model resnet50

# Use GPU
python demo.py --device cuda
```

The demo showcases:
- Standard inference
- MC Dropout uncertainty estimation
- Test-Time Augmentation (TTA)
- Adversarial robustness testing

### 3. Train a Model

```bash
# Train with default configuration
python train.py

# Train on specific dataset
python train.py --dataset chestmnist

# Custom training parameters
python train.py --model densenet121 --epochs 50 --batch_size 32 --lr 0.001

# Custom config file
python train.py --config configs/config.yaml
```

Best model checkpoints are saved to `checkpoints/`.

### 4. Evaluate Robustness

```bash
# Standard evaluation
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth

# With TTA
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth --tta

# Uncertainty analysis
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth --uncertainty

# Adversarial robustness testing
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth --robustness

# Full evaluation pipeline
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth --tta --uncertainty --robustness
```

Results are saved to `results/` including:
- Detailed metrics in JSON format
- ROC curve visualizations
- Uncertainty distribution plots
- Robustness comparison charts

---

## 🛡️ Robustness Modules

### 1. Deep Ensembles

**Paper**: *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (Lakshminarayanan et al., NeurIPS 2017)

Deep ensembles train multiple independently-initialized models and aggregate their predictions at inference time. This approach provides:

- **Improved accuracy** through model averaging (typically +1-3% AUC)
- **Uncertainty estimates** via inter-model disagreement
- **Better robustness** to noise and distribution shifts
- **Out-of-distribution detection** capability via increased prediction variance

**Usage**:
```python
from robust.deep_ensemble import DeepEnsemble

def create_model():
    return get_backbone('resnet50', num_classes=14)

ensemble = DeepEnsemble(create_model, num_models=5)
ensemble.create_models()

# Ensemble prediction with uncertainty
result = ensemble.predict(x)
print(f"Mean prediction: {result['probs']}")
print(f"Prediction std (uncertainty): {result['std_probs']}")
```

### 2. MC Dropout Uncertainty Estimation

**Paper**: *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning* (Gal & Ghahramani, ICML 2016)

Monte Carlo Dropout approximates Bayesian inference by keeping dropout enabled during inference. Multiple stochastic forward passes provide:

- **Epistemic uncertainty** — uncertainty due to model ignorance (reducible with more data)
- **Confidence calibration** — more reliable probability estimates
- **Clinical decision support** — flagging high-uncertainty cases for human review

**Applications in medical imaging**:
- Automatic triage (low uncertainty → AI confident, high uncertainty → human review)
- Active learning (prioritize uncertain cases for annotation)
- Out-of-distribution detection (OOD samples typically have higher uncertainty)

**Usage**:
```python
from robust.uncertainty import UncertaintyEstimator

estimator = UncertaintyEstimator(model, device='cuda')
result = estimator.mc_dropout(x, n_samples=30)

print(f"Mean prediction: {result['mean_probs']}")
print(f"Prediction std: {result['std_probs']}")
print(f"Predictive entropy: {result['entropy']}")
```

### 3. Test-Time Augmentation (TTA)

Test-Time Augmentation applies multiple transformations (flips, crops, rotations) to the same input image and aggregates predictions. This technique:

- **Improves accuracy** without retraining (typically +0.5-2%)
- **Enhances robustness** to spatial transformations
- **Provides stability** across imaging protocol variations
- **Computationally efficient** — trade compute for performance

**Usage**:
```python
from robust.tta import LightTTA, TestTimeAugmentation

# Lightweight TTA (horizontal flip only)
tta = LightTTA(model, device='cuda')
result = tta.predict(x)

# Full TTA with 6 augmentation strategies
full_tta = TestTimeAugmentation(model, image_size=224)
result = full_tta.predict(images)
```

### 4. Adversarial & Noise Robustness Testing

Evaluating model performance under adversarial attacks and noise perturbations is crucial for understanding model limitations.

**Supported attacks**:
- **FGSM** (Fast Gradient Sign Method) — single-step adversarial attack
- **PGD** (Projected Gradient Descent) — iterative strong attack
- **Gaussian Noise** — random noise perturbation
- **Salt-and-Pepper Noise** — pixel corruption simulation

**Why this matters for medical imaging**:
- Distribution shifts between hospitals/devices can act like "natural adversarial examples"
- Robustness testing reveals model fragilities before deployment
- Adversarial training can improve generalization

**Usage**:
```python
from robust.adversarial import AdversarialTester

tester = AdversarialTester(model, device='cuda')

# FGSM attack evaluation
result = tester.evaluate_robustness(x, y, attack_type='fgsm', epsilon=0.03)

# Full robustness benchmark
benchmark = tester.robustness_benchmark(x, y)
```

---

## 📊 Supported Datasets

| Dataset | Task | Classes | Train | Val | Test | Image Sizes |
|---------|------|---------|-------|-----|------|-------------|
| ChestMNIST | Multi-label | 14 | 73,126 | 10,000 | 25,536 | 28/64/128/224 |
| DermaMNIST | Multi-class | 7 | 7,007 | 1,003 | 2,005 | 28/64/128/224 |
| PneumoniaMNIST | Binary | 2 | 4,708 | 524 | 624 | 28/64/128/224 |

Datasets are automatically downloaded on first run via the `medmnist` package.

---

## ⚙️ Configuration

Key configuration options (`configs/config.yaml`):

```yaml
# Dataset configuration
dataset:
  name: "chestmnist"      # Dataset name
  image_size: 224         # Input image size
  in_channels: 1          # Input channels

# Model configuration
model:
  backbone: "resnet50"    # Backbone architecture
  pretrained: true        # ImageNet pretrained weights
  dropout_rate: 0.3       # Dropout rate (for MC Dropout)

# Training configuration
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 10

# Robustness configuration
robustness:
  deep_ensemble:
    enabled: true
    n_models: 5
  tta:
    enabled: true
  uncertainty:
    method: "mc_dropout"
    n_samples: 30
```

---

## 🎓 Academic Value & Citation

### Research Contributions

This framework provides a standardized pipeline for evaluating robustness and uncertainty in medical image classification. It can be used as:

- **Baseline** for robustness research in medical imaging
- **Educational tool** for teaching uncertainty estimation concepts
- **Benchmark platform** for comparing different robustness techniques
- **Starting point** for clinical AI projects requiring reliability guarantees

### Key Papers

If you use this framework in your research, please consider citing the following papers:

```bibtex
@inproceedings{lakshminarayanan2017simple,
  title={Simple and scalable predictive uncertainty estimation using deep ensembles},
  author={Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

@inproceedings{gal2016dropout,
  title={Dropout as a bayesian approximation: Representing model uncertainty in deep learning},
  author={Gal, Yarin and Ghahramani, Zoubin},
  booktitle={International conference on machine learning},
  year={2016}
}

@inproceedings{yang2021medmnist,
  title={Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis},
  author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  year={2021}
}

@inproceedings{goodfellow2015explaining,
  title={Explaining and harnessing adversarial examples},
  author={Goodfellow, Ian J and Shlens, Jonathon and Szegedy, Christian},
  booktitle={International Conference on Learning Representations},
  year={2015}
}
```

### Related Projects

- [MedMNIST](https://github.com/MedMNIST/MedMNIST) — The MedMNIST benchmark dataset collection
- [torchxrayvision](https://github.com/mlmed/torchxrayvision) — Chest X-ray library with pre-trained models
- [Albumentations](https://github.com/albumentations-team/albumentations) — Fast image augmentation library
- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch image models

---



---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It is NOT intended for clinical use or medical diagnosis. Medical image interpretation should always be performed by qualified healthcare professionals.

---

## 📧 Contact

ethan.link@foxmail.com

---

**If you find this project useful, please consider giving it a ⭐ Star!**
