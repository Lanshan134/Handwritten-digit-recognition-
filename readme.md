
# 🧠 MNIST 手写数字识别 —— 神经网络与 CNN 综合实验

本项目基于 Python，使用 NumPy 与 PyTorch 等工具，围绕 MNIST 手写数字识别任务，从最基本的 BP 神经网络出发，逐步完成改进、结构优化、参数调优等实验，最终构建出准确率超过 99% 的卷积神经网络（CNN）模型。

---

## 📁 文件结构说明

```
.
├── data.py                              # MNIST 数据加载（NumPy 版本）
├── basic_bp.py                          # 基本 BP 网络实现
├── bp_improved_1_loss_and_regularization.py  # 加入交叉熵与L2正则化的 BP 改进
├── bp_loss_compare_integrated.py        # MSE vs CrossEntropy 多次实验对比
├── bp_loss_sensitivity_experiment.py    # 不同超参数组合下的敏感性测试
├── bp_l2_search.py                      # L2 正则化强度搜索实验
├── cnn_mnist_compare.py                 # CNN中 ReLU vs Sigmoid 激活函数对比
├── cnn_mnist_optimizer_compare.py       # Adam vs SGD 优化器对比
├── cnn_hyperparam_search.py             # CNN中学习率与批量大小网格搜索
├── cnn_mnist_advanced.py                # 加入 BatchNorm + Dropout 的最终 CNN 模型
├── model/                               # 所有自动保存的模型参数（按准确率编号）
├── result/                              # 实验结果目录（csv、txt）
├── plot/                                # 训练曲线、热力图、柱状图等所有可视化图表
└── README.md                            # 项目说明（本文件）
```

---

## 🧪 实验列表与功能

| 实验编号 | 模块名（脚本）                                    | 核心功能                            |
| ---- | ------------------------------------------ | ------------------------------- |
| 1    | `basic_bp.py`                              | 基本的多层感知机（Sigmoid + MSE）实现       |
| 2    | `bp_improved_1_loss_and_regularization.py` | 引入交叉熵损失与 L2 正则化，对比性能变化          |
| 3    | `bp_loss_compare_integrated.py`            | 多随机种子重复实验，验证交叉熵更稳定              |
| 4    | `bp_l2_search.py`                          | L2 正则强度对准确率的影响测试                |
| 5    | `bp_loss_sensitivity_experiment.py`        | 不同 loss 对超参数敏感性分析               |
| 6    | `cnn_mnist_compare.py`                     | CNN 中使用 Sigmoid 与 ReLU 的效果对比    |
| 7    | `cnn_mnist_optimizer_compare.py`           | 对比 Adam 与 SGD 优化器的性能            |
| 8    | `cnn_hyperparam_search.py`                 | 学习率与批量大小的网格搜索与热力图可视化            |
| 9    | `cnn_mnist_advanced.py`                    | 加入 BatchNorm 与 Dropout 的高性能模型构建 |

---

## 🧠 主要技术点

* 神经网络基础：前向传播、误差反向传播（BP算法）、激活函数实现
* 深度学习训练：交叉熵损失、Softmax、梯度下降、L2正则化
* 模型调优：优化器（Adam/SGD）、超参数搜索（Grid Search）
* 模型正则化：Dropout、Batch Normalization
* 模型保存与重现：支持按准确率自动编号保存模型文件
* 可视化分析：准确率曲线、热力图、箱线图、柱状图、预测可视化

---

## 🔧 运行方式
项目运行环境如下：
- IDE：pycharm
- Python 3.11.0
- torch：2.6.0+cu126

确保环境包含如下依赖：

```bash
pip install numpy pandas matplotlib seaborn torch torchvision
```

运行示例（以CNN对比实验为例）：

```bash
python cnn_mnist_compare.py
```

模型将自动保存至 `model/` 文件夹，可视化图表生成至 `plot/`，结果数据存于 `result/`。

---

## 📌 项目亮点总结

* 兼具教学性与实用性，覆盖 BP 与 CNN 的完整训练流程；
* 多维度实验设置（损失函数、正则项、优化器、结构改进）；
* 自动保存与可视化支持，便于报告撰写与复现实验；
* 测试准确率最高达 **99.03%**（加入 BatchNorm + Dropout 的 CNN）；

---

## 📄 联系与参考

* 实验数据集：MNIST ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))
* 相关理论参考：Deep Learning (Ian Goodfellow)，CS231n (Stanford)
* 作者：张旭 @ 北京理工大学

---

