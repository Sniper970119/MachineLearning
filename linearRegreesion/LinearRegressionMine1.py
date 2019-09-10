# coding:utf-8
import pandas
import numpy
import matplotlib.pyplot as plt

# 首先读取数据
data = pandas.read_csv('./data/ex1data1.txt', header=None, names=['Population', 'Profit'])
# 绘制数据图像
# data.plot(kind='scatter', x='Population', y='Profit')
# plt.show()

# 数据预处理
# 增加一列数据，方便运算
data.insert(0, 'Ones', 1)
# 分割输入数据与输出数据，并转换为矩阵
col = data.shape[1]

"""
这里 注意不要漏了iloc 函数的调用
iloc函数为用下标分割数据行
loc函数为用索引分割数据行（即属性）
"""
X = numpy.mat(data.iloc[:, 0:col - 1].values)
Y = numpy.mat(data.iloc[:, col - 1:col].values)
# 初始化theta
theta = numpy.mat([0, 0])


def computeCost(X, Y, theta):
    """
    计算代价函数
    :param X: 输入数据
    :param Y: 输出数据
    :param theta: theta
    :return: 代价
    """
    temp = numpy.power((X * theta.T - Y), 2)
    return numpy.sum(temp) / (2 * len(X))


# 计算当前theta的代价函数
print(computeCost(X, Y, theta))


def grandientDescent(X, Y, theta, epoch=10000, alpha=0.01):
    """
    计算梯度下降的公式
    :param X: 输入数据
    :param Y: 输出数据
    :param theta: theta
    :param epoch: 训练批次
    :param alpha: 学习率
    :return: 训练后的theta以及代价变化列表
    """
    # 首先初始化一个代价训练列表，用来观察代价的变化情况
    cost = numpy.zeros(epoch)
    for i in range(epoch):
        theta = theta - (alpha / len(X)) * (X * theta.T - Y).T * X
        cost[i] = computeCost(X, Y, theta)
    return theta, cost


# 调用梯度下降算法
theta, cost = grandientDescent(X, Y, theta)
# 计算当前theta的代价函数
print(computeCost(X, Y, theta))

# 对计算出的theta进行画图
# 首先将横坐标分割100等份
"""
将数据进行等分 使用linspace方法
需要将绘图部分着重再看一看
"""
x = numpy.linspace(data.Population.min(), data.Population.max(), 100)
# 对纵坐标进行计算，使用 y = ax + b
y = (theta[0, 1] * x) + theta[0, 0]
# 绘图
fig, ax = plt.subplots(figsize=(6, 4))
# 绘制直线
ax.plot(x, y, 'r', label='Prediction')
# 绘制散列点
ax.scatter(data['Population'], data['Profit'], label='Training Data')
# 绘制右上的提示位
ax.legend(loc=2)
# 设置横坐标名称
ax.set_xlabel('Population')
# 设置纵坐标名称
ax.set_ylabel('Profit')
# 设置图名
ax.set_title('Predict Profit vs. Population Size')
# 绘图
plt.show()
