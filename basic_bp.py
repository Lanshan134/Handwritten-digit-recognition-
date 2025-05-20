import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import load_mnist, one_hot  # 依赖 data.py

# 超参数
input_size = 784
hidden_size = 64
output_size = 10
learning_rate = 0.1
epochs = 10
batch_size = 64

# 创建结果目录
os.makedirs("result/basic_bp", exist_ok=True)
os.makedirs("plot/basic_bp", exist_ok=True)
os.makedirs("model/basic_bp", exist_ok=True)
# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# 数据加载
X_train, y_train_raw, X_test, y_test_raw = load_mnist()
y_train = one_hot(y_train_raw)
y_test = one_hot(y_test_raw)

# 网络参数初始化
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

train_acc_list = []
train_loss_list = []
train_time_list = []

# 训练过程
for epoch in range(epochs):
    start_time = time.time()
    permutation = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[permutation], y_train[permutation]

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward
        z1 = np.dot(X_batch, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss (MSE)
        loss = np.mean((a2 - y_batch) ** 2)

        # Backward
        dz2 = (a2 - y_batch) * sigmoid_derivative(a2)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
        dW1 = np.dot(X_batch.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 更新参数
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # 每轮训练完评估训练集
    z1_full = np.dot(X_train, W1) + b1
    a1_full = sigmoid(z1_full)
    z2_full = np.dot(a1_full, W2) + b2
    a2_full = sigmoid(z2_full)
    epoch_loss = np.mean((a2_full - y_train) ** 2)
    epoch_acc = accuracy(a2_full, y_train)
    epoch_time = time.time() - start_time

    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)
    train_time_list.append(epoch_time)

    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {epoch_time:.2f}s")

# 测试集评估
z1_test = np.dot(X_test, W1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, W2) + b2
a2_test = sigmoid(z2_test)
test_acc = accuracy(a2_test, y_test)

# 保存指标结果
metrics = pd.DataFrame({
    "Epoch": list(range(1, epochs + 1)),
    "Train Accuracy": train_acc_list,
    "Train Loss": train_loss_list,
    "Train Time (s)": train_time_list
})
metrics.to_csv("result/basic_bp/metrics.csv", index=False)
with open("result/basic_bp/test_accuracy.txt", "w") as f:
    f.write(f"Final Test Accuracy: {test_acc:.4f}\n")

# 可视化准确率
plt.figure()
plt.plot(metrics["Epoch"], metrics["Train Accuracy"], label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train Accuracy")
plt.grid(True)
plt.savefig("plot/basic_bp/accuracy.png")

# 可视化损失
plt.figure()
plt.plot(metrics["Epoch"], metrics["Train Loss"], label="Train Loss", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss")
plt.grid(True)
plt.savefig("plot/basic_bp/loss.png")

# 可视化时间
plt.figure()
plt.plot(metrics["Epoch"], metrics["Train Time (s)"], label="Train Time", color='green')
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.title("Training Time per Epoch")
plt.grid(True)
plt.savefig("plot/basic_bp/time.png")

# 混淆矩阵分析
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(a2_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("plot/basic_bp/confusion_matrix.png")

# 展示部分预测图像，展示预测标签与真实标签
# 合并展示前 10 张测试图像，显示预测与真实标签
plt.figure(figsize=(12, 4))  # 控制整张图大小
for i in range(10):
    plt.subplot(2, 5, i + 1)  # 2行5列的子图
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"T:{np.argmax(y_test[i])} P:{y_pred[i]}", fontsize=10)
    plt.axis('off')

plt.suptitle("Prediction vs Ground Truth (First 10 Samples)")
plt.tight_layout()
plt.savefig("plot/basic_bp/sample_predictions.png")  # 保存为大图
plt.show()


# 自动编号保存模型（避免覆盖）
model_dir = "model/basic_bp"
os.makedirs(model_dir, exist_ok=True)
base_name = f"basic_bp_model"
acc_str = f"{test_acc:.4f}"
i = 1
while os.path.exists(os.path.join(model_dir, f"{base_name}_{i}_{acc_str}.npz")):
    i += 1
model_path = os.path.join(model_dir, f"{base_name}_{i}_{acc_str}.npz")
np.savez(model_path, W1=W1, b1=b1, W2=W2, b2=b2)
print(f"模型已保存到: {model_path}")


print(f"Final Test Accuracy: {test_acc:.4f} saved to result/basic_bp/test_accuracy.txt")
