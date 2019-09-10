# coding:utf-8

import pandas
import numpy
import matplotlib.pyplot as plt

# 首先获得数据
data = pandas.read_csv('./data/ex1data2.txt', header=None, names=['Size', 'BedRoom', 'Price'])
# 数据标准化
"""
数据标准化方法，（数据 - 平均数）/ 标准差  
数据标准化需要在插入新数据之前进行,否则由于Ones列的加入，会让数据的平均值和标准差变0（因为一列均为1）
"""
data = (data - data.mean()) / data.std()
# 插入数据
data.insert(0, 'Ones', 1)
# 先部分初始化数据，试试画三维散点图
cols = data.shape[1]
x = data.iloc[:, 0:1].values
y = data.iloc[:, 1:2].values
z = data.iloc[:, 2:3].values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()

# 获取输入数据和输出数据
X = numpy.mat(data.iloc[:, 0:3].values)
Y = numpy.mat(data.iloc[:, 3:4].values)
theta = numpy.mat([0, 0, 0])


def computeCost(X, Y, theta):
    """
    计算代价函数
    :param X: 输入数据
    :param Y: 目标数据
    :param theta: theta
    :return: 代价
    """
    temp = numpy.power(((X * theta.T) - Y), 2)
    return numpy.sum(temp) / (2 * len(X))


# 显示当前的代价 （	0.48936170212765967
print(computeCost(X, Y, theta))


def gradientDescent(X, Y, theta, epoch=10000, alpha=0.01):
    """
    梯度下降算法
    :param X: 输入数据
    :param Y: 目标数据
    :param theta: theta
    :param epoch: 训练批次
    :param alpha: 学习率
    :return: 代价列表以及theta
    """
    cost = numpy.zeros(epoch)
    for i in range(epoch):
        theta = theta - (alpha / len(X)) * ((X * theta.T) - Y).T * X
        cost[i] = computeCost(X, Y, theta)
    return cost, theta


cost, theta = gradientDescent(X, Y, theta)
# 输出当前的代价 （0.13068648053904197
print(computeCost(X, Y, theta))

# 绘制图像(虽然好像二维以上并没有什么意义
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制散列点
ax.scatter(x, y, z, label='data')
# 计算横坐标
x = numpy.linspace(data['Size'].min(), data['Size'].max(), 100)
y = numpy.linspace(data['BedRoom'].min(), data['BedRoom'].max(), 100)
# z = ax + by + c
z = theta[0, 0] + theta[0, 1] * x + theta[0, 2] * y

ax.plot(x, y, z, 'r', label='Prediction')
ax.legend(loc=2)
plt.show()
