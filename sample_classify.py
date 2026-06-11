import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.metrics import accuracy_score

# ===================== 1. 路径&防卡死配置（已按你真实路径修正） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 你确认的真实路径
DATA_ROOT = "/kaggle/input/datasets/organizations/nih-chest-xrays/sample"
IMG_DIR = os.path.join(DATA_ROOT, "sample", "images")
LABEL_CSV = os.path.join(DATA_ROOT, "sample", "sample_labels.csv")

print(f"图片目录: {IMG_DIR}")
print(f"标签文件: {LABEL_CSV}")

# 防卡死核心参数
BATCH_SIZE = 4
EPOCHS = 3
TRAIN_RATIO = 0.8
MODEL_SAVE_PATH = "/kaggle/working/pneumonia_resnet18.pth"
NUM_WORKERS = 0

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===================== 2. 适配你数据的数据集（带真实标签） =====================
class CXRDataset(Dataset):
    def __init__(self, img_dir, label_csv, transform=None, is_train=True, train_ratio=0.8):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

        # 1. 读取标签表
        self.df = pd.read_csv(label_csv)
        print(f"加载标签表，总样本数: {len(self.df)}")

        # 2. 遍历匹配图片 + 生成标签（沿用你原本的标签规则）
        all_pairs = []
        for _, row in self.df.iterrows():
            img_name = row["Image Index"]
            label_text = row["Finding Labels"].lower()
            img_path = os.path.join(img_dir, img_name)

            # 标签：normal/negative=0，其余（含肺炎）=1
            label = 0 if label_text in ["normal", "negative"] else 1

            if os.path.exists(img_path):
                all_pairs.append((img_path, label))

        print(f"有效图片总数: {len(all_pairs)}")
        # 随机打乱（保证划分随机性）
        random.shuffle(all_pairs)

        # 3. 划分训练/验证集
        split_idx = int(len(all_pairs) * train_ratio)
        if self.is_train:
            self.data = all_pairs[:split_idx]
        else:
            self.data = all_pairs[split_idx:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ===================== 3. 加载数据 =====================
train_dataset = CXRDataset(IMG_DIR, LABEL_CSV, transform=transform, is_train=True)
val_dataset = CXRDataset(IMG_DIR, LABEL_CSV, transform=transform, is_train=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False
)

# ===================== 4. 加载模型 =====================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 二分类
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===================== 5. 训练&验证函数 =====================
def train_one_epoch():
    model.train()
    running_loss = 0.0
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (idx + 1) % 10 == 0:
            print(f"训练批次 {idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    return running_loss / len(train_loader)

def val_one_epoch():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)

# ===================== 6. 启动训练 =====================
print("\n===== 开始训练 =====")
for epoch in range(EPOCHS):
    print(f"\n---------- Epoch {epoch+1}/{EPOCHS} ----------")
    train_loss = train_one_epoch()
    val_acc = val_one_epoch()
    print(f"本轮平均损失: {train_loss:.4f} | 验证集准确率: {val_acc:.4f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ 训练完成！模型已保存至: {MODEL_SAVE_PATH}")