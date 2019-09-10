# -*- coding:utf-8 -*-

import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def loadData():
    """
    读取数据
    :return:
    """
    mig = io.imread('data/bird_small.png')
    plt.imshow(mig)
    return mig / 255.


def initcentroids(X, K):
    """
    随机初始化分类中心点
    :param X:
    :param K:
    :return:
    """
    m, n = np.shape(X)
    idx = np.random.choice(m, K)
    return X[idx]


def findClosestCentroids(X, centroids):
    """
    依据分类中心点划分数据点
    :param X:
    :param centroids:
    :return:
    """
    idx = []
    max_distance = 100000
    for i in range(len(X)):
        distance = X[i] - centroids
        euclid_distance = distance[:, 0] ** 2 + distance[:, 1] ** 2
        if euclid_distance.min() < max_distance:
            ci = np.argmin(euclid_distance)
            idx.append(ci)
    return np.array(idx)


def computeCentroids(X, idx):
    """
    计算新的中心分类点
    :param X:
    :param centroids:
    :return:
    """
    centroids = []
    for i in range(len(np.unique(idx))):
        u_k = X[idx == i].mean(axis=0)
        centroids.append(u_k)
    return np.array(centroids)


def runKMeans(X, K, max_iters):
    """
    执行k 均值算法
    :param X:
    :param K:
    :param max_iters:
    :return:
    """
    centroids = initcentroids(X, K)
    centroids_all = []
    centroids_all.append(centroids)
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids_i = computeCentroids(X, idx)
        centroids_all.append(centroids_i)
    return idx, centroids_all


if __name__ == '__main__':
    A = loadData()
    X = A.reshape(-1, 3)
    idx, centroids_all = runKMeans(X, 16, 10)
    img = np.zeros(X.shape)
    centroids = centroids_all[-1]
    for i in range(len(centroids)):
        img[idx == i] = centroids[i]

    img = img.reshape((128, 128, 3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(A)
    axes[1].imshow(img)
    plt.show()
