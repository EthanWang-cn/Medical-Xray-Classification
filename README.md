# 医学影像鲁棒性分类项目 (Med-Robust-CV)

> 基于 PyTorch 的医学影像分类框架，集成多种鲁棒性增强技术，适用于胸片、皮肤镜等医学影像分析场景。

## 📋 项目简介

本项目是一个面向医学影像分类的鲁棒性深度学习框架，基于 **MedMNIST** 数据集构建，支持胸部X光、皮肤镜等多种医学影像模态。项目不仅实现了基础的分类功能，更重点集成了多种**鲁棒性增强技术**，包括深度集成学习、不确定性估计、测试时增强和对抗鲁棒性测试，旨在提升医学影像AI模型在真实临床场景下的可靠性和可信度。

### 🎯 核心特性

- **多数据集支持**：ChestMNIST（胸片14类疾病）、DermaMNIST（皮肤病变7类）、PneumoniaMNIST（肺炎检测）
- **多种骨干网络**：ResNet、DenseNet、EfficientNet 等主流架构
- **鲁棒性增强**：
  - ✅ **深度集成学习** (Deep Ensembles) - 多模型投票提升准确率
  - ✅ **不确定性估计** (MC Dropout) - 量化预测可信度
  - ✅ **测试时增强** (TTA) - 提升预测稳定性
  - ✅ **对抗鲁棒性测试** - FGSM/PGD攻击评估模型脆弱性
  - ✅ **噪声鲁棒性测试** - 高斯噪声、椒盐噪声
- **完整工具链**：训练、评估、可视化、Demo推理一站式支持
- **详细中文注释**：代码注释详尽，适合学习和二次开发

## 🏗️ 项目结构

```
med-robust-cv/
├── README.md              # 项目说明文档
├── requirements.txt       # Python 依赖包
├── .gitignore             # Git 忽略配置
├── configs/
│   └── config.yaml        # 配置文件（数据集、模型、训练参数等）
├── data/
│   └── README.md          # 数据集说明
├── models/
│   ├── __init__.py
│   ├── backbone.py        # 骨干网络实现（ResNet/DenseNet/EfficientNet）
│   └── robust_module.py   # 鲁棒性分类器封装
├── robust/
│   ├── __init__.py
│   ├── deep_ensemble.py   # 深度集成学习
│   ├── tta.py             # 测试时增强 (TTA)
│   ├── uncertainty.py     # 不确定性估计
│   └── adversarial.py     # 对抗攻击与鲁棒性测试
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # 评估指标（AUC/F1/精确率/召回率等）
│   └── visualization.py   # 可视化工具（ROC曲线、混淆矩阵等）
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
└── demo.py                # Demo 推理脚本
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git init  # 你可以直接在当前目录初始化 git
# 或 git clone <your-repo-url>

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行 Demo

无需训练，直接运行 Demo 体验鲁棒性功能：

```bash
# 运行 Demo（使用随机图像演示）
python demo.py

# 使用自己的图像
python demo.py --image path/to/your/image.png

# 指定数据集和模型
python demo.py --dataset chestmnist --model resnet50

# 指定设备
python demo.py --device cuda
```

Demo 会演示以下功能：
- 标准预测
- MC Dropout 不确定性估计
- 测试时增强 (TTA)
- 对抗鲁棒性测试

### 3. 训练模型

```bash
# 使用默认配置训练
python train.py

# 指定数据集
python train.py --dataset chestmnist

# 指定模型和训练参数
python train.py --model densenet121 --epochs 50 --batch_size 32 --lr 0.001

# 使用自定义配置文件
python train.py --config configs/config.yaml
```

训练完成后，最佳模型会保存在 `checkpoints/` 目录下。

### 4. 评估模型

```bash
# 标准评估
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth

# 启用 TTA 评估
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth --tta

# 启用不确定性分析
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth --uncertainty

# 启用鲁棒性测试
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth --robustness

# 全部启用
python evaluate.py --checkpoint checkpoints/best_model_chestmnist.pth --tta --uncertainty --robustness
```

评估结果会保存在 `results/` 目录下，包括：
- 详细的指标 JSON 文件
- ROC 曲线图
- 不确定性分布图
- 鲁棒性对比图

## 📊 鲁棒性模块详解

### 1. 深度集成学习 (Deep Ensembles)

**原理**：训练多个独立的模型，推理时对预测结果进行平均或投票。

**优势**：
- 显著提升分类准确率（通常 1-3%）
- 提供基于模型分歧的不确定性估计
- 对噪声和扰动更鲁棒

**使用方式**：
```python
from robust.deep_ensemble import DeepEnsemble

# 创建集成模型
def create_model():
    return get_backbone('resnet50', num_classes=14)

ensemble = DeepEnsemble(create_model, num_models=5)
ensemble.create_models()

# 集成预测
result = ensemble.predict(x)
print(f"集成预测概率: {result['probs']}")
print(f"预测不确定性: {result['std_probs']}")
```

### 2. 不确定性估计 (Uncertainty Estimation)

**方法**：蒙特卡洛 Dropout (MC Dropout)

**原理**：在推理时保持 Dropout 开启，进行多次前向传播，利用预测的方差估计认知不确定性。

**应用场景**：
- 识别模型"不知道"的样本
- 临床决策支持（高不确定性样本建议人工复核）
- 主动学习（选择不确定性高的样本标注）

**使用方式**：
```python
from robust.uncertainty import UncertaintyEstimator

estimator = UncertaintyEstimator(model, device='cuda')
result = estimator.mc_dropout(x, n_samples=30)

print(f"预测均值: {result['mean_probs']}")
print(f"预测标准差: {result['std_probs']}")
print(f"预测熵: {result['entropy']}")
```

### 3. 测试时增强 (Test-Time Augmentation, TTA)

**原理**：对同一张测试图像应用多种变换（翻转、裁剪、旋转等），然后集成所有变换的预测结果。

**优势**：
- 无需重新训练即可提升性能
- 增强对图像平移、旋转、缩放的鲁棒性
- 实现简单，计算开销可控

**使用方式**：
```python
from robust.tta import LightTTA, TestTimeAugmentation

# 轻量级 TTA（仅水平翻转）
tta = LightTTA(model, device='cuda')
result = tta.predict(x)

# 完整 TTA（6种变换）
full_tta = TestTimeAugmentation(model, image_size=224)
result = full_tta.predict(images)
```

### 4. 对抗鲁棒性测试

**支持的攻击方法**：
- **FGSM** (Fast Gradient Sign Method) - 快速单步攻击
- **PGD** (Projected Gradient Descent) - 迭代式强攻击
- **高斯噪声** - 随机噪声扰动
- **椒盐噪声** - 像素损坏模拟

**用途**：
- 评估模型的鲁棒性
- 发现模型的脆弱点
- 指导对抗训练

**使用方式**：
```python
from robust.adversarial import AdversarialTester

tester = AdversarialTester(model, device='cuda')

# FGSM 攻击
result = tester.evaluate_robustness(x, y, attack_type='fgsm', epsilon=0.03)

# 完整鲁棒性基准测试
benchmark = tester.robustness_benchmark(x, y)
```

## 📈 支持的数据集

| 数据集 | 任务类型 | 类别数 | 训练集 | 验证集 | 测试集 | 图像尺寸 |
|--------|----------|--------|--------|--------|--------|----------|
| ChestMNIST | 多标签分类 | 14 | 73,126 | 10,000 | 25,536 | 28/64/128/224 |
| DermaMNIST | 多分类 | 7 | 7,007 | 1,003 | 2,005 | 28/64/128/224 |
| PneumoniaMNIST | 二分类 | 2 | 4,708 | 524 | 624 | 28/64/128/224 |

数据集会在首次运行时自动下载，无需手动准备。

## 🛠️ 配置说明

主要配置项（`configs/config.yaml`）：

```yaml
# 数据集配置
dataset:
  name: "chestmnist"      # 数据集名称
  image_size: 224         # 图像尺寸
  in_channels: 1          # 输入通道数

# 模型配置
model:
  backbone: "resnet50"    # 骨干网络
  pretrained: true        # 预训练权重
  dropout_rate: 0.3       # Dropout 比率

# 训练配置
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 10

# 鲁棒性配置
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

## 📝 代码亮点

1. **模块化设计**：各功能模块独立，易于扩展和维护
2. **详细中文注释**：每个函数、类都有详细的文档字符串和注释
3. **配置驱动**：通过 YAML 配置文件管理超参数
4. **完整的评估体系**：AUC、F1、精确率、召回率等多维度评估
5. **丰富的可视化**：ROC 曲线、混淆矩阵、不确定性分布等

## 🎓 学习资源

### 相关论文

- **Deep Ensembles**: *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (NeurIPS 2017)
- **MC Dropout**: *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning* (ICML 2016)
- **TTA**: *Test-Time Augmentation for Image Classification*
- **FGSM**: *Explaining and Harnessing Adversarial Examples* (ICLR 2015)
- **PGD**: *Towards Deep Learning Models Resistant to Adversarial Attacks* (ICLR 2018)

### 参考项目

- [MedMNIST](https://github.com/MedMNIST/MedMNIST) - 医学影像基准数据集
- [torchxrayvision](https://github.com/mlmed/torchxrayvision) - 胸部X光工具库
- [Albumentations](https://github.com/albumentations-team/albumentations) - 数据增强库

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## ⚠️ 免责声明

本项目仅供学术研究和学习使用，**不应用于临床诊断或医疗决策**。医学影像的解读需要由专业的医疗人员进行。

## 📧 联系方式

如有问题或建议，欢迎通过 Issue 联系。

---

**如果这个项目对你有帮助，欢迎给个 Star ⭐**
