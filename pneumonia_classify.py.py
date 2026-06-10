import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score

# ===================== 1. 配置参数 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")
BATCH_SIZE = 16
EPOCHS = 5  # 少轮次训练，加快跑通
DATA_ROOT = "./chest_xray"  # 数据集路径

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===================== 2. 自定义数据集 =====================
class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 0: NORMAL  1: PNEUMONIA
        for label, cls_name in enumerate(["NORMAL", "PNEUMONIA"]):
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ===================== 3. 加载数据 =====================
train_dataset = PneumoniaDataset(os.path.join(DATA_ROOT, "train"), transform)
val_dataset = PneumoniaDataset(os.path.join(DATA_ROOT, "val"), transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 4. 加载预训练模型 =====================
model = models.resnet18(pretrained=True)
# 替换最后一层全连接层，适配二分类
fc_in_features = model.fc.in_features
model.fc = nn.Linear(fc_in_features, 2)
model.to(DEVICE)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===================== 5. 训练流程 =====================
def train_one_epoch():
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# 验证流程
def val_one_epoch():
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc

# 开始训练
if __name__ == "__main__":
    print("开始训练...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch()
        val_acc = val_one_epoch()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "pneumonia_resnet18.pth")
    print("模型已保存为 pneumonia_resnet18.pth")