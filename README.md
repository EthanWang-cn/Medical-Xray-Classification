# Medical X-ray Pneumonia Classification
基于PyTorch+ResNet18实现胸部X光肺炎检测，医疗影像AI入门Demo。

## 项目简介
本项目使用经典卷积神经网络ResNet18，对胸部X光影像进行二分类任务（正常影像 / 肺炎影像），完整实现**数据加载、图像预处理、模型训练、验证、推理**全流程。
- 技术栈：Python / PyTorch / TorchVision
- 任务类型：医学图像二分类（医疗计算机视觉）
- 运行环境：CPU/GPU 均可

## 数据集
数据集：ChestX-Ray Pneumonia
下载地址：https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## 运行方式
1. 安装依赖
```bash
pip install torch torchvision matplotlib pillow scikit-learn