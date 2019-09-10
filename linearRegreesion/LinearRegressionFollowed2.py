# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data/ex1data2.txt", names=['Size', 'Bedrooms', 'Price'])
plt.show()

# 进行归一化特征 mean 获取平均值  std获取标准差
# 这一步是将很大的数据映射到一个很小的范围并且保留数据的特征，防止了梯度爆炸
data = (data - data.mean()) / data.std()

# 首先进行数据划分，增肌Ones列
data.insert(0, "Ones", 1)
# 获取列数
cols = data.shape[1]
# 分割输入向量并转换为矩阵
inputVector = data.iloc[:, 0:cols - 1]
inputVector = np.mat(inputVector)
# 获取目标向量并转换为矩阵
aimVector = data.iloc[:, cols - 1:cols]
aimVector = np.mat(aimVector)
# 定义theta
theta = np.mat([0, 0, 0])


def computeCost(inputVector, aimVector, theta):
    """
    计算代价函数
    :param inputVector: 输入向量
    :param aimVector: 目标向量
    :param theta: theta
    :return:
    """
    inner = np.power(((inputVector * theta.T) - aimVector), 2)
    return np.sum(inner) / 2 * len(inputVector)


# 当前代价 （1081.0000000000002
nowCost = computeCost(inputVector, aimVector, theta)

# 定义学习率
alpha = 0.01
# 定义迭代次数
epoch = 1000


def gradientDescent(inputVector, aimVector, theta, alpha, epoch):
    # 代价数组，记录代价变化
    cost = np.zeros(epoch)
    length = len(inputVector)
    for i in range(epoch):
        theta = theta - (alpha / length) * ((inputVector * theta.T) - aimVector).T * inputVector
        cost[i] = computeCost(inputVector, aimVector, theta)
    return theta, cost


finalTheta, costList = gradientDescent(inputVector, aimVector, theta, alpha, epoch)
# 当前的代价 （288.7237434634511
nowCost = computeCost(inputVector, aimVector, finalTheta)

# 绘制代价变化图
fig, ax = plt.subplots()
ax.plot(np.arange(epoch), costList, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


# 使用正规方程来计算theta
def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


finalTheta2 = normalEqn(inputVector, aimVector)  # 感觉和批量梯度下降的theta的值有点差距

