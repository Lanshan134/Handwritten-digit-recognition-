import os
import time
import numpy as np
import pandas as pd
from data import load_mnist, one_hot
from basic_bp import sigmoid, sigmoid_derivative, accuracy

# 超参数组合列表
configs = [
    {"loss": "mse", "lr": 0.01, "hidden": 64, "batch": 64},
    {"loss": "mse", "lr": 0.1,  "hidden": 64, "batch": 64},
    {"loss": "mse", "lr": 0.1,  "hidden": 128, "batch": 64},
    {"loss": "mse", "lr": 0.1,  "hidden": 64, "batch": 128},
    {"loss": "crossentropy", "lr": 0.01, "hidden": 64, "batch": 64},
    {"loss": "crossentropy", "lr": 0.1,  "hidden": 64, "batch": 64},
    {"loss": "crossentropy", "lr": 0.1,  "hidden": 128, "batch": 64},
    {"loss": "crossentropy", "lr": 0.1,  "hidden": 64, "batch": 128},
]

seeds = [42, 52, 62]  # 每组实验重复3次

os.makedirs("result/bp_loss_sensitivity", exist_ok=True)

def train_model(loss_type, lr, hidden_size, batch_size, seed):
    input_size, output_size = 784, 10
    np.random.seed(seed)

    X_train, y_train_raw, X_test, y_test_raw = load_mnist()
    y_train = one_hot(y_train_raw)
    y_test = one_hot(y_test_raw)

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    for epoch in range(10):
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

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

    # 测试准确率
    a1_test = sigmoid(np.dot(X_test, W1) + b1)
    z2_test = np.dot(a1_test, W2) + b2
    if loss_type == "mse":
        a2_test = sigmoid(z2_test)
    else:
        exp_scores = np.exp(z2_test - np.max(z2_test, axis=1, keepdims=True))
        a2_test = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return accuracy(a2_test, one_hot(y_test_raw))

# 跑所有组合
records = []
for cfg in configs:
    for seed in seeds:
        acc = train_model(cfg["loss"], cfg["lr"], cfg["hidden"], cfg["batch"], seed)
        records.append({
            "loss": cfg["loss"],
            "lr": cfg["lr"],
            "hidden": cfg["hidden"],
            "batch": cfg["batch"],
            "seed": seed,
            "test_acc": acc
        })
        print(f'{cfg["loss"]:>10} | lr={cfg["lr"]} | h={cfg["hidden"]} | b={cfg["batch"]} | acc={acc:.4f}')

# 保存CSV
df = pd.DataFrame(records)
df.to_csv("result/bp_loss_sensitivity/summary.csv", index=False)
print("所有超参数组合测试完成，结果已保存。")
