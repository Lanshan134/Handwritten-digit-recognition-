import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data import load_mnist, one_hot
from basic_bp import sigmoid, sigmoid_derivative, accuracy

# 参数设置
input_size = 784
hidden_size = 64
output_size = 10
learning_rate = 0.1
epochs = 10
batch_size = 64
seeds = [42, 52, 62, 72, 82]

# 创建目录
os.makedirs("result/bp_loss_repeat", exist_ok=True)
os.makedirs("plot/bp_loss_repeat", exist_ok=True)

def train_model(loss_type, seed):
    np.random.seed(seed)
    X_train, y_train_raw, X_test, y_test_raw = load_mnist()
    y_train = one_hot(y_train_raw)
    y_test = one_hot(y_test_raw)

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    acc_list = []
    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[permutation], y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            z1 = np.dot(X_batch, W1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, W2) + b2
            if loss_type == "mse":
                a2 = sigmoid(z2)
                dz2 = (a2 - y_batch) * sigmoid_derivative(a2)
            else:
                exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                dz2 = (a2 - y_batch) / batch_size

            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
            dW1 = np.dot(X_batch.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        a1_full = sigmoid(np.dot(X_train, W1) + b1)
        z2_full = np.dot(a1_full, W2) + b2
        if loss_type == "mse":
            a2_full = sigmoid(z2_full)
        else:
            exp_scores = np.exp(z2_full - np.max(z2_full, axis=1, keepdims=True))
            a2_full = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        acc = accuracy(a2_full, y_train)
        acc_list.append(acc)

    # 测试准确率
    a1_test = sigmoid(np.dot(X_test, W1) + b1)
    z2_test = np.dot(a1_test, W2) + b2
    if loss_type == "mse":
        a2_test = sigmoid(z2_test)
    else:
        exp_scores = np.exp(z2_test - np.max(z2_test, axis=1, keepdims=True))
        a2_test = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    test_acc = accuracy(a2_test, y_test)

    return acc_list, test_acc

def run_and_visualize():
    result = {"mse": [], "crossentropy": []}
    test_accuracies = {"mse": [], "crossentropy": []}

    for seed in seeds:
        acc_mse, test_mse = train_model("mse", seed)
        acc_ce, test_ce = train_model("crossentropy", seed)
        result["mse"].append(acc_mse)
        result["crossentropy"].append(acc_ce)
        test_accuracies["mse"].append(test_mse)
        test_accuracies["crossentropy"].append(test_ce)

    df_mse = pd.DataFrame(result["mse"]).T
    df_ce = pd.DataFrame(result["crossentropy"]).T
    df_final = pd.DataFrame(test_accuracies)
    df_final.to_csv("result/bp_loss_repeat/test_accuracy_summary.csv", index=False)

    # 图1：训练曲线均值±标准差
    plt.figure()
    plt.plot(df_mse.mean(axis=1), label="MSE", color='blue')
    plt.fill_between(range(epochs), df_mse.mean(axis=1) - df_mse.std(axis=1),
                     df_mse.mean(axis=1) + df_mse.std(axis=1), alpha=0.2, color='blue')
    plt.plot(df_ce.mean(axis=1), label="CrossEntropy", color='orange')
    plt.fill_between(range(epochs), df_ce.mean(axis=1) - df_ce.std(axis=1),
                     df_ce.mean(axis=1) + df_ce.std(axis=1), alpha=0.2, color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy (mean ± std)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot/bp_loss_repeat/accuracy_mean_std.png")

    # 图2：每轮准确率叠加图
    plt.figure()
    for run in result["mse"]:
        plt.plot(run, color="blue", alpha=0.3)
    for run in result["crossentropy"]:
        plt.plot(run, color="orange", alpha=0.3)
    plt.title("Training Accuracy Overlay")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("plot/bp_loss_repeat/accuracy_overlay.png")

    # 图3：柱状图
    plt.figure()
    avg_acc = df_final.mean()
    bars = plt.bar(avg_acc.index, avg_acc.values, color=["skyblue", "orange"])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom')
    plt.ylabel("Test Accuracy")
    plt.title("Average Test Accuracy")
    plt.grid(True, axis='y')
    plt.savefig("plot/bp_loss_repeat/test_accuracy_bar.png")

    # 图4：箱线图
    plt.figure()
    df_melted = df_final.melt(var_name="Loss Type", value_name="Test Accuracy")
    sns.boxplot(data=df_melted, x="Loss Type", y="Test Accuracy", palette="Set2")
    sns.stripplot(data=df_melted, x="Loss Type", y="Test Accuracy", color='black', size=6, jitter=True)
    plt.title("Test Accuracy Distribution (Boxplot)")
    plt.grid(True)
    plt.savefig("plot/bp_loss_repeat/test_accuracy_boxplot.png")

    print("全部训练与图像保存完成")

if __name__ == "__main__":
    run_and_visualize()
