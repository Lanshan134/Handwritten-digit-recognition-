import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("model/CNN", exist_ok=True)
os.makedirs("result/cnn_optimizer_compare", exist_ok=True)
os.makedirs("plot/cnn_optimizer_compare", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

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

def train_and_evaluate(optimizer_name='adam', epochs=5):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer: choose 'adam' or 'sgd'")

    train_accs = []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
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

        train_acc = correct / total
        train_accs.append(train_acc)
        print(f"[{optimizer_name}] Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc:.4f}")

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
    print(f"[{optimizer_name}] Final Test Accuracy: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"model/CNN/cnn_{optimizer_name}.pth")
    return train_accs, test_acc

if __name__ == "__main__":
    optimizers = ['adam', 'sgd']
    results = {}
    for opt in optimizers:
        train_accs, test_acc = train_and_evaluate(opt)
        results[opt] = {'train_accs': train_accs, 'test_acc': test_acc}

    # 画训练准确率曲线
    plt.figure()
    for opt in optimizers:
        plt.plot(range(1, len(results[opt]['train_accs']) + 1), results[opt]['train_accs'], label=opt)
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("plot/cnn_optimizer_compare/train_accuracy_compare.png")

    # 测试准确率柱状图
    plt.figure()
    bars = plt.bar(optimizers, [results[opt]['test_acc'] for opt in optimizers])
    plt.ylim(0, 1.0)
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.4f}", ha='center', va='bottom')
    plt.grid(True, axis='y')
    plt.savefig("plot/cnn_optimizer_compare/test_accuracy_compare.png")

    # 保存结果
    import pandas as pd
    df = pd.DataFrame({
        'optimizer': optimizers,
        'test_accuracy': [results[opt]['test_acc'] for opt in optimizers]
    })
    df.to_csv("result/cnn_optimizer_compare/test_accuracy.csv", index=False)

