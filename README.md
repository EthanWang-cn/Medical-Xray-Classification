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
```

标量形式： 
$$ \begin{align*} y_1 &= w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1 \\ y_2 &= w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2 \end{align*} $$
等价于

$$ \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} $$

 展开计算： 
 $$ \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} w_{11}x_1 + w_{12}x_2 + w_{13}x_3 \\ w_{21}x_1 + w_{22}x_2 + w_{23}x_3 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} $$
 
 最终： 
 $$ \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1 \\ w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2 \end{bmatrix} $$
 矩阵表示

 $$ Y = g(WX+b)$$
抽象后得到神经网络前向传播的通用公式

 $$ A^{[L]} = g\left( W^{[L]} A^{[L-1]} + b^{[L]} \right) $$
- A[L]：第 L 层的激活值（输出）
- W[L]：第 L 层的权重矩阵
- b[L]：第 L 层的偏置向量
- A[L−1]：上一层（第 L−1 层）的激活值（输入）

- g：激活函数（如 sigmoid、ReLU 等）