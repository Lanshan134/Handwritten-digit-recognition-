import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 创建模型保存目录
os.makedirs("model/CNN", exist_ok=True)
os.makedirs("result/cnn_compare", exist_ok=True)
os.makedirs("plot/cnn_compare", exist_ok=True)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# CNN模型定义
class SimpleCNN(nn.Module):
    def __init__(self, activation='relu'):
        super(SimpleCNN, self).__init__()
        act_fn = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            act_fn,
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            act_fn,
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            act_fn,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 训练与测试函数
def train_and_evaluate(activation):
    model = SimpleCNN(activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    train_accs = []

    for epoch in range(epochs):
        correct, total = 0, 0
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        train_accs.append(acc)
        print(f"[{activation}] Epoch {epoch+1}/{epochs} - Train Accuracy: {acc:.4f}")

    # 测试
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print(f"[{activation}] Final Test Accuracy: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"model/CNN/cnn_{activation}.pth")

    return train_accs, test_acc

# 运行两组实验
relu_accs, relu_test_acc = train_and_evaluate('relu')
sigmoid_accs, sigmoid_test_acc = train_and_evaluate('sigmoid')

# 绘图
plt.figure()
plt.plot(range(1, 6), relu_accs, label='ReLU')
plt.plot(range(1, 6), sigmoid_accs, label='Sigmoid')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.title('Train Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.savefig("plot/cnn_compare/train_accuracy_compare.png")

# 测试准确率柱状图
import pandas as pd
result_df = pd.DataFrame({
    "Activation": ["ReLU", "Sigmoid"],
    "Test Accuracy": [relu_test_acc, sigmoid_test_acc]
})
result_df.to_csv("result/cnn_compare/test_accuracy.csv", index=False)

plt.figure()
bars = plt.bar(["ReLU", "Sigmoid"], [relu_test_acc, sigmoid_test_acc])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", ha='center', va='bottom')
plt.title("Test Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.grid(True, axis='y')
plt.savefig("plot/cnn_compare/test_accuracy_compare.png")
