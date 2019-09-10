# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def loadData():
    """
    读取数据
    :return:
    """
    mat = sio.loadmat('data/ex7data2.mat')
    X = mat['X']
    return X, mat


def findClosestControids(X, centroids):
    """
    寻找里分类中心点最近的点集合
    :param X: 原始数据
    :param centroids: 分类中心点
    :return: 分类结果组成的一维数组
    """
    idx = []
    max_distance = 1000000  # 最大距离
    for i in range(len(X)):
        # 该点和各个点的距离
        distances = X[i] - centroids
        # 计算欧几里德距离
        euclid_distance = distances[:,0] ** 2 + distances[:,1] ** 2
        if euclid_distance.min() < max_distance:
            ci = np.argmin(euclid_distance)
            idx.append(ci)
    return np.array(idx)


def computeCentroids(X, idx):
    """
    计算分类中心
    :param X: 原始数据
    :param idx: 一维分类结果数组
    :return: 计算后的中心点
    """
    centroids = []
    for i in range(len(np.unique(idx))):
        # 计算每一列的平均值
        u_k = X[idx == i].mean(axis=0)
        centroids.append(u_k)
    return np.array(centroids)


def runKmeans(X, centroids, max_iters):
    """
    k均值算法，
    :param X: 原始数据
    :param centroids: 中心点
    :param max_iters: 最大迭代次数
    :return: 执行后的分类结果以及中心点变化轨迹
    """
    centroid_all = []
    centroid_all.append(centroids)
    idx = []
    for i in range(max_iters):
        idx = findClosestControids(X, centroids)
        centroid_i = computeCentroids(X, idx)
        centroid_all.append(centroid_i)
    return idx, centroid_all


def initCentroids(X, K):
    """
    随机初始化类中心点
    :param X: 原始数据
    :param K: 分k
    :return: 找到的中心点
    """
    m, n = np.shape(X)
    centroid_index = np.random.choice(m, K)
    centroids = X[centroid_index]
    return centroids


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


if __name__ == '__main__':
    # 读取数据
    X, _ = loadData()
    # 执行三次
    for i in range(3):
        # 随机初始化
        centroids = initCentroids(X, 3)
        idx, centroids_all = runKmeans(X, centroids, 10)
        plotData(X, centroids_all, idx)
        pass
