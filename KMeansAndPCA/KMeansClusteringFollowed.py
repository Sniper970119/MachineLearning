# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def findClosestCentroids(X, centroids):
    """
    output a one-dimensional array idx that holds the
    index of the closest centroid to every training example.
    输出一个一维数组idx，其中包含最接近每个分类的中心的点集合。
    :param X: 数据点
    :param centroids: 分类中心点
    :return:
    """
    idx = []
    max_dist = 1000000  # 限制一下最大距离
    for i in range(len(X)):
        minus = X[i] - centroids  # 该点与三个分类中心的距离
        dist = minus[:, 0] ** 2 + minus[:, 1] ** 2  # 计算欧几里德距离
        if dist.min() < max_dist:  # 限制最大距离
            ci = np.argmin(dist)  # 寻找最小距离的点
            idx.append(ci)
    return np.array(idx)


mat = loadmat('data/ex7data2.mat')
# print(mat.keys())
X = mat['X']
init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosestCentroids(X, init_centroids)
# [0 2 1]
print(idx[0:3])


def computeCentroids(X, idx):
    """
    计算新的分类中心点
    :param X: 数据点
    :param idx: 分类的一维数组
    :return:
    """
    centroids = []
    for i in range(len(np.unique(idx))):  # 遍历每一个k
        u_k = X[idx == i].mean(axis=0)  # 求每列的平均值
        centroids.append(u_k)
    return np.array(centroids)


"""
[[2.42830111 3.15792418]
 [5.81350331 2.63365645]
 [7.11938687 3.6166844 ]]
"""
print(computeCentroids(X, idx))


def plotData(X, centroids, idx=None):
    """
    可视化数据，并自动分开着色。
    :param X: 原始数据
    :param centroids: 包含每次中心点历史记录
    :param idx: 最后一次迭代生成的idx向量，存储每个样本分配的簇中心点的值
    :return:
    """
    # 颜色列表
    colors = ['b', 'g', 'gold', 'darkorange', 'salmon', 'olivedrab', 'maroon', 'navy', 'sienna', 'tomato', 'lightgray',
              'gainsboro', 'coral', 'aliceblue', 'dimgray', 'mintcream', 'mintcream']
    # 断言，如果条件为真就正常运行，条件为假就出发异常
    assert len(centroids[0]) <= len(colors), 'colors not enough '

    subX = []  # 分号类的样本点
    if idx is not None:
        for i in range(centroids[0].shape[0]):
            x_i = X[idx == i]
            subX.append(x_i)

    else:
        subX = [X]  # 将X转化为一个元素的列表，每个元素为每个簇的样本集，方便下方绘图

    # 分别画出每个簇的点，并着不同的颜色
    plt.figure(figsize=(8, 5))
    for i in range(len(subX)):
        xx = subX[i]
        plt.scatter(xx[:, 0], xx[:, 1], c=colors[i], label='Cluster %d' % i)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    plt.title('Plot of X Points', fontsize=16)

    # 画出簇中心点的移动轨迹
    xx, yy = [], []
    for centroid in centroids:
        xx.append(centroid[:, 0])
        yy.append(centroid[:, 1])

    plt.plot(xx, yy, 'rx--', markersize=8)
    plt.show()


plotData(X, [init_centroids])


def runKmeans(X, centroids, max_iters):
    """
    执行k-均值算法
    :param X:  原始数据
    :param centroids: 中心点
    :param max_iters:最大迭代次数
    :return:
    """
    # k  ，虽然没啥用
    K = len(centroids)
    # 所有的点（多次迭代）
    centroids_all = []
    centroids_all.append(centroids)
    centroid_i = centroids
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroid_i)
        centroid_i = computeCentroids(X, idx)
        centroids_all.append(centroid_i)
    return idx, centroids_all


idx, centroids_all = runKmeans(X, init_centroids, 20)
plotData(X, centroids_all, idx)


def initCentroids(X, K):
    """
    随机初始化
    :param X:
    :param K:
    :return:
    """
    m, n = X.shape
    idx = np.random.choice(m, K)  # 随机从m中选择k个
    centroids = X[idx]

    return centroids


# 随机寻找三个随机初始中心
for i in range(3):
    centroids = initCentroids(X, 3)
    idx, centroids_all = runKmeans(X, centroids, 10)
    plotData(X, centroids_all, idx)
