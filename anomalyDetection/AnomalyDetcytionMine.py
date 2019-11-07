# -*- coding:utf-8 -*-

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_data():
    """
    读取数据
    :return:
    """
    mat = loadmat('data/ex8data1.mat')
    X = mat['X']
    Xval, yval = mat['Xval'], mat['yval']
    return X, Xval, yval


def get_gaussian_garams(X, useMultivariate=False):
    """
    确定高斯模型参数
    :param X: 数据
    :param useMultivariate: 是否使用多元高斯模型
    :return: 高斯参数的mu和sigma^2
    """
    mu = X.mean(axis=0)
    if useMultivariate:
        sigma2 = ((X - mu).T @ (X - mu)) / len(X)
    else:
        sigma2 = X.var(axis=0, ddof=0)
    return mu, sigma2


def gaussian(X, mu, sigma2):
    """
    获取数据通过高斯模型之后的映射
    :param X:
    :param mu:
    :param sigma2:
    :return:
    """
    m, n = X.shape
    # 如果sigma2是一维的，转换多维方便计算
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)
    # 计算公式
    norm = 1. / (np.power((2 * np.pi), n / 2) * np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m, 1))
    for row in range(m):
        xrow = X[row]
        exp[row] = np.exp(-0.5 * ((xrow - mu).T).dot(np.linalg.inv(sigma2)).dot(xrow - mu))
    return norm * exp


def select_threshold(yval, pval):
    """
    选择一个合适的阈值作为异常数据的检测边界
    :param yval: 数据结果（key）
    :param pval: 通过高斯映射后的值
    :return:
    """

    def computeF1(yval, pval):
        """
        计算F1
        :param yval:
        :param pval:
        :return:
        """
        m = len(yval)
        tp = float(len([i for i in range(m) if pval[i] and yval[i]]))
        fp = float(len([i for i in range(m) if pval[i] and not yval[i]]))
        fn = float(len([i for i in range(m) if not pval[i] and yval[i]]))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return F1

    # 分割数据
    epslions = np.linspace(min(pval), max(pval), 1000)
    bestF1, bestEpslion = 0, 0
    # 寻找一个最合适的点，来作为最佳的阈值
    for e in epslions:
        _pvel = pval < e
        thisF1 = computeF1(yval, _pvel)
        if thisF1 > bestF1:
            bestF1 = thisF1
            bestEpslion = e
    return bestF1, bestEpslion


def plot_data(X):
    """
    绘制数据点，由于需要多次绘制，使用方法简化流程
    :return:
    """
    plt.figure(figsize=(8, 5))
    plt.plot(X[:, 0], X[:, 1], 'bx')


def plot_contours(mu, sigma2):
    """
    画出高斯分布图
    :param mu:
    :param sigma2:
    :return:
    """
    delta = 0.3
    # 分别在0，30中间均分出x y
    x = np.arange(0, 30, delta)
    y = np.arange(0, 30, delta)
    # 将x y转换为坐标
    xx, yy = np.meshgrid(x, y)
    points = np.c_[xx.ravel(), yy.ravel()]
    z = gaussian(points, mu, sigma2)
    z = z.reshape(xx.shape)
    cont_levels = [10 ** h for h in range(-20, 0, 3)]
    plt.contour(xx, yy, z, cont_levels)
    plt.title('Gaussian Contours', fontsize=16)


X, Xval, yval = load_data()
mu, sigma2 = get_gaussian_garams(X, useMultivariate=True)
pval = gaussian(Xval, mu, sigma2)
bestF1, bestEpsilon = select_threshold(yval, pval)

y = gaussian(X, mu, sigma2)  # X的概率
# 计算离群点
xx = np.array([X[i] for i in range(len(y)) if y[i] < bestEpsilon])
plot_data(X)
plot_contours(mu, sigma2)
# 绘制离群点
plt.scatter(xx[:, 0], xx[:, 1], s=80, facecolors='none', edgecolors='r')
plt.show()
print()
