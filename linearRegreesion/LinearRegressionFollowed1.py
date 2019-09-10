# -*- coding:utf-8 -*-
"""
线性回归，根据提示完成。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义数据集
dataPath = './data/ex1data1.txt'
# names添加列名，header用指定的行来作为标题，若原无标题且指定标题则设为None
data = pd.read_csv(dataPath, header=None, names=["Population", 'Profit'])


# 画图
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8, 5))
# 显示图像
plt.show()


def computeCost(x, y, theta):
    """
    计算代价函数
    :param x: 输入向量
    :param y: 目标向量
    :param theta: theta
    :return:
    """
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))


# 在数据中添加一个Ones列，方便我们计算代价和梯度，三个参数分别为，插入位置、插入列名、默认值
# 这样在切割掉目标向量之后输入向量依然保持在2列，方便与 未知数 矩阵相乘（因为未知数有2个）
data.insert(0, 'Ones', 1)
# 获取数据
# 获取列数
cols = data.shape[1]
# 取前cols-1列，即输入向量
x = data.iloc[:, 0:cols - 1]
# 取最后一列，即目标向量
y = data.iloc[:, cols - 1:cols]

# 分别将两组数据转换成矩阵形式，方便运算
x = np.mat(x.values)
y = np.mat(y.values)
# 初始化theta(这个值为最终分割线的 [截距，斜率]
theta = np.mat([0, 0])
# 计算代价（32.072733877455676）
nowCost = computeCost(x, y, theta)


def gradientDescent(x, y, theta, alpha, epoch):
    """
    梯度下降
    :param x: x轴数据
    :param y: y轴数据
    :param theta: theta
    :param alpha: 学习率
    :param epoch:θ
    :return:
    """
    # 参数 θ的数量，用来绘制训练次数和代价的关系
    cost = np.zeros(epoch)
    # 输入数据的大小
    m = x.shape[0]
    # 进行epoch次迭代
    for i in range(epoch):
        # 利用向量化一步求解
        theta = theta - (alpha / m) * (x * theta.T - y).T * x
        # 保存当前的代价到数组中
        cost[i] = computeCost(x, y, theta)
    return theta, cost


# 学习率
alpha = 0.01
# 计算次数
epoch = 1000
final_theta, cost = gradientDescent(x, y, theta, alpha, epoch)
# 显示当前代价 （4.515955503078912
nowCost = computeCost(x, y, final_theta)

# 在data.Population的最大最小值范围内返回均匀间隔的数字（100个 # 横坐标
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# 纵坐标
f = (final_theta[0, 1] * x) + final_theta[0, 0]
fig, ax = plt.subplots(figsize=(6, 4))
# 绘制直线  (横坐标  纵坐标 颜色  名字
ax.plot(x, f, 'r', label='Prediction')
# 绘制点   ( 横坐标  纵坐标  名字
ax.scatter(data['Population'], data.Profit, label='Traning Data')
# 标签显示位置  2表示在左上角
ax.legend(loc=2)
# 设置横坐标名称
ax.set_xlabel('Population')
# 设置纵坐标名称
ax.set_ylabel('Profit')
# 设置标题
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
# 绘制直线  arange 和range功能类似 均分数据
ax.plot(np.arange(epoch), cost, 'r')
# 设置横纵坐标以及图片名称
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

# data.mean(axis=0) 输出矩阵为一行,求每列的平均值,同理data.mean(axis=1) 输出矩阵为一列,求每行的平均值
# data.std(axis=0) 输出矩阵为一列,求每列的标准差,同理data.std(axis=1) 输出矩阵为一列,求每行的标准差


print()
