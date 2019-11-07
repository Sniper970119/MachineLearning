# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

mat = loadmat('data/ex8data1.mat')
# dict_keys(['__header__', '__version__', '__globals__', 'X', 'Xval', 'yval'])
print(mat.keys())
X = mat['X']
Xval, yval = mat['Xval'], mat['yval']
# ((307, 2), (307, 2), (307, 1))
print(X.shape, Xval.shape, yval.shape)


def plot_data():
    """
    绘制数据样式，第一次输出
    :return:
    """
    plt.figure(figsize=(8, 5))
    # 训练集
    plt.plot(X[:, 0], X[:, 1], 'bx')
    # 验证集
    # plt.scatter(Xval[:,0], Xval[:,1], c=yval.flatten(), marker='x', cmap='rainbow')


def gaussian(X, mu, sigma2):
    '''
    mu, sigma2参数已经决定了一个高斯分布模型
    因为原始模型就是多元高斯模型在sigma2上是对角矩阵而已，所以如下：
    If Sigma2 is a matrix, it is treated as the covariance matrix.
    If Sigma2 is a vector, it is treated as the sigma^2 values of the variances
    in each dimension (a diagonal covariance matrix)
    output:
        一个(m, )维向量，包含每个样本的概率值。
    '''
    m, n = X.shape
    # ndim返回的是数组的维度,这里将一维转换为多维，方便计算
    if np.ndim(sigma2) == 1:
        # diag：array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
        #       array是一个二维矩阵时，结果输出矩阵的对角线元素
        # 这里输出了对角线矩阵
        sigma2 = np.diag(sigma2)

    # 计算参数
    norm = 1. / (np.power((2 * np.pi), n / 2) * np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m, 1))
    for row in range(m):
        xrow = X[row]
        exp[row] = np.exp(-0.5 * ((xrow - mu).T).dot(np.linalg.inv(sigma2)).dot(xrow - mu))
    return norm * exp


def getGaussianParams(X, useMultivariate):
    """
    获取高斯参数
    :param X: data
    :param useMultivariate: 是否使用多维高斯函数
    :return: mu和参数
    """
    # 求平均值，mu
    mu = X.mean(axis=0)
    if useMultivariate:
        # 多维高斯参数计算方法  有四个参数 2*2
        sigma2 = ((X - mu).T @ (X - mu)) / len(X)
    else:
        # 样本方差 只有两个参数
        sigma2 = X.var(axis=0, ddof=0)

    return mu, sigma2


def plotContours(mu, sigma2):
    """
    画出高斯概率分布的图，在三维中是一个上凸的曲面。投影到平面上则是一圈圈的等高线。
    """
    # 计算点的间隔
    delta = .3  # 注意delta不能太小！！！否则会生成太多的数据，导致矩阵相乘会出现内存错误。
    x = np.arange(0, 30, delta)
    y = np.arange(0, 30, delta)

    # 这部分要转化为X形式的坐标矩阵，也就是一列是横坐标，一列是纵坐标，
    # 然后才能传入gaussian中求解得到每个点的概率值
    # meshgrid 将数据转换为坐标形式
    xx, yy = np.meshgrid(x, y)
    # 是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
    points = np.c_[xx.ravel(), yy.ravel()]  # 按列合并，一列横坐标，一列纵坐标
    z = gaussian(points, mu, sigma2)
    z = z.reshape(xx.shape)  # 这步骤不能忘

    # 这个levels是作业里面给的参考,或者通过求解的概率推出来。
    cont_levels = [10 ** h for h in range(-20, 0, 3)]
    plt.contour(xx, yy, z, cont_levels)

    plt.title('Gaussian Contours', fontsize=16)


# First contours without using multivariate gaussian:
plot_data()
useMV = False
plotContours(*getGaussianParams(X, useMV))
plt.show()

# Then contours with multivariate gaussian:
plot_data()
useMV = True
plotContours(*getGaussianParams(X, useMV))
plt.show()
print()


def selectThreshold(yval, pval):
    """
    选择合适的阈值，即多小的值应该被判断为异常
    :param yval:
    :param pval:
    :return:
    """

    def computeF1(yval, pval):
        m = len(yval)
        tp = float(len([i for i in range(m) if pval[i] and yval[i]]))
        fp = float(len([i for i in range(m) if pval[i] and not yval[i]]))
        fn = float(len([i for i in range(m) if not pval[i] and yval[i]]))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return F1

    # 数据分割 1000份
    epsilons = np.linspace(min(pval), max(pval), 1000)
    bestF1, bestEpsilon = 0, 0
    for e in epsilons:
        pval_ = pval < e
        thisF1 = computeF1(yval, pval_)
        if thisF1 > bestF1:
            bestF1 = thisF1
            bestEpsilon = e

    return bestF1, bestEpsilon


# 获取mu 和sigma
mu, sigma2 = getGaussianParams(X, useMultivariate=True)
# 利用mu和sigma确定高斯函数（使用验证集）
pval = gaussian(Xval, mu, sigma2)
# 确定最佳参数
bestF1, bestEpsilon = selectThreshold(yval, pval)
# (0.8750000000000001, 8.999852631901397e-05)

# 确定高斯函数（使用全部数据）
y = gaussian(X, mu, sigma2)  # X的概率
xx = np.array([X[i] for i in range(len(y)) if y[i] < bestEpsilon])
print(xx)  # 离群点

plot_data()
plotContours(mu, sigma2)
plt.scatter(xx[:, 0], xx[:, 1], s=80, facecolors='none', edgecolors='r')
plt.show()
print()


#
mat = loadmat('data/ex8data2.mat')
X2 = mat['X']
Xval2, yval2 = mat['Xval'], mat['yval']
print(X2.shape)  # (1000, 11)

mu, sigma2 = getGaussianParams(X2, useMultivariate=False)
ypred = gaussian(X2, mu, sigma2)

yval2pred = gaussian(Xval2, mu, sigma2)

# You should see a value epsilon of about 1.38e-18, and 117 anomalies found.
bestF1, bestEps = selectThreshold(yval2, yval2pred)
anoms = [X2[i] for i in range(X2.shape[0]) if ypred[i] < bestEps]
# (1.378607498200024e-18, 117)
print(bestEps, len(anoms))

plt.show()
print()
