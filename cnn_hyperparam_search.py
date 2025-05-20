import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("model/CNN", exist_ok=True)
os.makedirs("result/cnn_hyperparam_search", exist_ok=True)
os.makedirs("plot/cnn_hyperparam_search", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_evaluate(lr, batch_size, epochs=5):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    print(f"lr={lr}, batch_size={batch_size}, Test Accuracy={test_acc:.4f}")
    return test_acc

if __name__ == "__main__":
    lrs = [0.001, 0.005, 0.01]
    batch_sizes = [32, 64, 128]
    results = []

    for lr in lrs:
        for bs in batch_sizes:
            acc = train_evaluate(lr, bs)
            results.append({"learning_rate": lr, "batch_size": bs, "test_accuracy": acc})

    df = pd.DataFrame(results)
    df.to_csv("result/cnn_hyperparam_search/search_results.csv", index=False)

    # 画热力图
    pivot = df.pivot("batch_size", "learning_rate", "test_accuracy")
    plt.figure(figsize=(8,6))
    plt.title("Test Accuracy for Learning Rate and Batch Size")
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.savefig("plot/cnn_hyperparam_search/hyperparam_heatmap.png")
    plt.show()
