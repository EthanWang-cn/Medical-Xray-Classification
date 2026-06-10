import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "pneumonia_resnet18.pth"  # 训练好的模型路径
CLASS_NAMES = ["正常胸片 (NORMAL)", "肺炎胸片 (PNEUMONIA)"]

# 图像预处理（和训练时保持一致！）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载模型
def load_model():
    model = models.resnet18(pretrained=False)
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 单张图片推理
def predict_image(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # 增加batch维度

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, confidence, img

if __name__ == "__main__":
    # 1. 加载模型
    model = load_model()
    print("模型加载完成！")

    # 2. 替换成你自己的图片路径
    image_path = "test_xray.jpg"  # 把这张图放在和脚本同级目录

    # 3. 推理
    pred_idx, confidence, img = predict_image(image_path, model)
    pred_class = CLASS_NAMES[pred_idx]

    # 4. 打印结果
    print(f"预测结果: {pred_class}")
    print(f"置信度: {confidence:.4f}")

    # 5. 可视化（加分项）
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(f"预测: {pred_class}\n置信度: {confidence:.2%}")
    plt.axis("off")
    plt.show()