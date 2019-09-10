# -*- coding:utf-8 -*-

from skimage import io
import numpy as np
import matplotlib.pyplot as plt


def initCentroids(X, K):
    """随机初始化"""
    m, n = X.shape
    idx = np.random.choice(m, K)
    centroids = X[idx]

    return centroids


def runKmeans(X, centroids, max_iters):
    K = len(centroids)
    centroids_all = []
    centroids_all.append(centroids)
    centroid_i = centroids
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroid_i)
        centroid_i = computeCentroids(X, idx)
        centroids_all.append(centroid_i)

    return idx, centroids_all


def findClosestCentroids(X, centroids):
    """
    output a one-dimensional array idx that holds the
    index of the closest centroid to every training example.
    """
    idx = []
    max_dist = 1000000  # 限制一下最大距离
    for i in range(len(X)):
        minus = X[i] - centroids  # here use numpy's broadcasting
        dist = minus[:, 0] ** 2 + minus[:, 1] ** 2
        if dist.min() < max_dist:
            ci = np.argmin(dist)
            idx.append(ci)
    return np.array(idx)


def computeCentroids(X, idx):
    centroids = []
    for i in range(len(np.unique(idx))):  # np.unique() means K
        u_k = X[idx == i].mean(axis=0)  # 求每列的平均值
        centroids.append(u_k)

    return np.array(centroids)


A = io.imread('data/bird_small.png')
# (128, 128, 3)
print(A.shape)
plt.imshow(A)
plt.show()
# Divide by 255 so that all values are in the range 0 - 1
A = A / 255.

#  Reshape the image into an (N,3) matrix where N = number of pixels.
#  Each row will contain the Red, Green and Blue pixel values
#  This gives us our dataset matrix X that we will use K-Means on.

X = A.reshape(-1, 3)
K = 16
centroids = initCentroids(X, K)
idx, centroids_all = runKmeans(X, centroids, 10)


img = np.zeros(X.shape)
centroids = centroids_all[-1]
for i in range(len(centroids)):
    img[idx == i] = centroids[i]

img = img.reshape((128, 128, 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(A)
axes[1].imshow(img)
plt.show()
