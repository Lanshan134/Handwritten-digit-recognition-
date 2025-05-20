import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import load_mnist, one_hot
from basic_bp import sigmoid, sigmoid_derivative, accuracy

# Softmax 与交叉熵
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# 模型基类
class BPModel:
    def __init__(self, name, loss="mse", reg=False, lambda_reg=0.01):
        self.name = name
        self.loss_type = loss
        self.use_reg = reg
        self.lambda_reg = lambda_reg
        self.input_size = 784
        self.hidden_size = 64
        self.output_size = 10
        self.learning_rate = 0.1
        self.epochs = 25
        self.batch_size = 64
        self.init_params()

    def init_params(self):
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        if self.loss_type == "mse":
            self.a2 = sigmoid(self.z2)
        else:
            self.a2 = softmax(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        if self.loss_type == "mse":
            loss = np.mean((y_pred - y_true) ** 2)
        else:
            loss = cross_entropy(y_pred, y_true)
        if self.use_reg:
            loss += self.lambda_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return loss

    def backward(self, X, y):
        m = X.shape[0]
        if self.loss_type == "mse":
            dz2 = (self.a2 - y) * sigmoid_derivative(self.a2)
        else:
            dz2 = (self.a2 - y) / m
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        if self.use_reg:
            dW2 += 2 * self.lambda_reg * self.W2
            dW1 += 2 * self.lambda_reg * self.W1

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X_train, y_train):
        acc_list, loss_list, time_list = [], [], []
        for epoch in range(self.epochs):
            start_time = time.time()
            permutation = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[permutation], y_train[permutation]

            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            y_pred_full = self.forward(X_train)
            loss = self.compute_loss(y_pred_full, y_train)
            acc = accuracy(y_pred_full, y_train)
            acc_list.append(acc)
            loss_list.append(loss)
            time_list.append(time.time() - start_time)

            print(f"[{self.name}] Epoch {epoch + 1:2d} | Loss: {loss:.4f} | Acc: {acc:.4f}")

        return acc_list, loss_list, time_list

    def evaluate(self, X_test, y_test):
        y_pred = self.forward(X_test)
        return accuracy(y_pred, y_test), y_pred

# 加载数据
X_train, y_train_raw, X_test, y_test_raw = load_mnist()
y_train = one_hot(y_train_raw)
y_test = one_hot(y_test_raw)

# 三组实验模型
models = [
    BPModel("A_mse", loss="mse", reg=False),
    BPModel("B_crossentropy", loss="crossentropy", reg=False),
    BPModel("C_crossentropy_reg", loss="crossentropy", reg=True, lambda_reg=0.0001),
]

# 存储指标
history = {}
for model in models:
    accs, losses, times = model.train(X_train, y_train)
    test_acc, _ = model.evaluate(X_test, y_test)
    history[model.name] = {
        "acc": accs, "loss": losses, "time": times, "test_acc": test_acc
    }
    # 自动保存模型，避免覆盖
    model_dir = "model/bp_improved"
    os.makedirs(model_dir, exist_ok=True)
    base_name = f"{model.name}_model"
    acc_str = f"{test_acc:.4f}"
    i = 1
    while os.path.exists(os.path.join(model_dir, f"{base_name}_{i}_{acc_str}.npz")):
        i += 1
    model_path = os.path.join(model_dir, f"{base_name}_{i}_{acc_str}.npz")
    np.savez(model_path, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    print(f"[{model.name}] 模型已保存到: {model_path}")

    print(f"[{model.name}] Final Test Accuracy: {test_acc:.4f}")

# 可视化
os.makedirs("plot/bp_improved", exist_ok=True)

plt.figure()
for name in history:
    plt.plot(history[name]["acc"], label=name)
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("plot/bp_improved/accuracy_compare.png")

plt.figure()
for name in history:
    plt.plot(history[name]["loss"], label=name)
plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("plot/bp_improved/loss_compare.png")

plt.figure()
for name in history:
    plt.plot(history[name]["time"], label=name)
plt.title("Train Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True)
plt.savefig("plot/bp_improved/time_compare.png")

# 保存每组实验数据与测试准确率
result_dir = "result/bp_improved_1"
os.makedirs(result_dir, exist_ok=True)

for name, record in history.items():
    df = pd.DataFrame({
        "Epoch": list(range(1, len(record["acc"]) + 1)),
        "Train Accuracy": record["acc"],
        "Train Loss": record["loss"],
        "Train Time (s)": record["time"],
    })
    df.to_csv(f"{result_dir}/{name}_metrics.csv", index=False)
    with open(f"{result_dir}/{name}_test_accuracy.txt", "w") as f:
        f.write(f"Final Test Accuracy: {record['test_acc']:.4f}\n")

# 绘制最终测试准确率柱状图
# 绘制最终测试准确率柱状图，带数据标签
plt.figure()
model_names = list(history.keys())
test_accuracies = [history[name]["test_acc"] for name in model_names]
bars = plt.bar(model_names, test_accuracies, color=['blue', 'orange', 'green'])

plt.title("Final Test Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.grid(True, axis='y')

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10)

plt.savefig("plot/bp_improved/test_accuracy_compare.png")
