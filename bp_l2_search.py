import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import load_mnist, one_hot
from basic_bp import sigmoid, sigmoid_derivative, accuracy

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class BPModel:
    def __init__(self, lambda_reg):
        self.lambda_reg = lambda_reg
        self.input_size = 784
        self.hidden_size = 64
        self.output_size = 10
        self.learning_rate = 0.1
        self.epochs = 10
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
        self.a2 = softmax(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        loss = cross_entropy(y_pred, y_true)
        loss += self.lambda_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return loss

    def backward(self, X, y):
        m = X.shape[0]
        dz2 = (self.a2 - y) / m
        dW2 = np.dot(self.a1.T, dz2) + 2 * self.lambda_reg * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) + 2 * self.lambda_reg * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X_train, y_train):
        for epoch in range(self.epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[permutation], y_train[permutation]
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)

    def evaluate(self, X_test, y_test):
        y_pred = self.forward(X_test)
        return accuracy(y_pred, y_test)

X_train, y_train_raw, X_test, y_test_raw = load_mnist()
y_train = one_hot(y_train_raw)
y_test = one_hot(y_test_raw)

lambda_list = [0, 0.0001, 0.001, 0.01, 0.1]
test_accuracies = []

for lambda_reg in lambda_list:
    print(f"Training with L2 Regularization λ = {lambda_reg}")
    model = BPModel(lambda_reg)
    model.train(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)
    test_accuracies.append(test_acc)
    print(f"Test Accuracy: {test_acc:.4f}")

os.makedirs("result/bp_l2_search", exist_ok=True)
result_df = pd.DataFrame({
    "Lambda": lambda_list,
    "Test Accuracy": test_accuracies
})
result_df.to_csv("result/bp_l2_search/l2_search_results.csv", index=False)

os.makedirs("plot/bp_l2_search", exist_ok=True)
plt.figure()
bars = plt.bar([str(l) for l in lambda_list], test_accuracies)
plt.title("L2 Regularization Strength vs Test Accuracy")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1.0)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", ha='center', va='bottom')
plt.grid(True, axis='y')
plt.savefig("plot/bp_l2_search/l2_test_accuracy_compare.png")
