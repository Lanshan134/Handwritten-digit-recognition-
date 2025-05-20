import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 创建输出目录
os.makedirs("result/cnn_advanced", exist_ok=True)
os.makedirs("plot/cnn_advanced", exist_ok=True)

# 改进版 CNN 模型（带 BatchNorm 和 Dropout）
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# 模型训练函数
def train(model, device, train_loader, optimizer, criterion, epochs=5):
    model.train()
    acc_list = []
    for epoch in range(epochs):
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        acc = correct / total
        acc_list.append(acc)
        print(f"Epoch {epoch+1}: Train Accuracy = {acc:.4f}")
    return acc_list

# 测试函数
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

# 主流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_acc = train(model, device, train_loader, optimizer, criterion)
    test_acc = test(model, device, test_loader)

    # 保存模型
    torch.save(model.state_dict(), f"result/cnn_advanced/model_acc_{test_acc:.4f}.pt")

    # 保存训练准确率图
    plt.figure()
    plt.plot(range(1, len(train_acc)+1), train_acc, label="Train Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy with BatchNorm + Dropout")
    plt.grid(True)
    plt.savefig("plot/cnn_advanced/train_accuracy.png")

    with open("result/cnn_advanced/test_accuracy.txt", "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

if __name__ == "__main__":
    main()
