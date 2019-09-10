# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat('data/ex7data1.mat')
X = mat['X']
# (50, 2)
print(X.shape)
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')


def featureNormalize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    X_norm = (X - means) / stds
    return X_norm, means, stds


def pca(X):
    sigma = (X.T @ X) / len(X)
    U, S, V = np.linalg.svd(sigma)

    return U, S, V


X_norm, means, stds = featureNormalize(X)
U, S, V = pca(X_norm)

print(U[:, 0])
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')

plt.plot([means[0], means[0] + 1.5 * S[0] * U[0, 0]],
         [means[1], means[1] + 1.5 * S[0] * U[0, 1]],
         c='r', linewidth=3, label='First Principal Component')
plt.plot([means[0], means[0] + 1.5 * S[1] * U[1, 0]],
         [means[1], means[1] + 1.5 * S[1] * U[1, 1]],
         c='g', linewidth=3, label='Second Principal Component')
plt.grid()
# changes limits of x or y axis so that equal increments of x and y have the same length
# 不然看着不垂直，不舒服。：）
plt.axis("equal")
plt.legend()
plt.show()


def projectData(X, U, K):
    Z = X @ U[:, :K]

    return Z


# project the first example onto the first dimension
# and you should see a value of about 1.481
Z = projectData(X_norm, U, 1)
print(Z[0])


def recoverData(Z, U, K):
    X_rec = Z @ U[:, :K].T

    return X_rec


# you will recover an approximation of the first example and you should see a value of
# about [-1.047 -1.047].
X_rec = recoverData(Z, U, 1)
print(X_rec[0])

plt.figure(figsize=(7,5))
plt.axis("equal")
plot = plt.scatter(X_norm[:,0], X_norm[:,1], s=30, facecolors='none',
                   edgecolors='b',label='Original Data Points')
plot = plt.scatter(X_rec[:,0], X_rec[:,1], s=30, facecolors='none',
                   edgecolors='r',label='PCA Reduced Data Points')

plt.title("Example Dataset: Reduced Dimension Points Shown",fontsize=14)
plt.xlabel('x1 [Feature Normalized]',fontsize=14)
plt.ylabel('x2 [Feature Normalized]',fontsize=14)
plt.grid(True)

for x in range(X_norm.shape[0]):
    plt.plot([X_norm[x,0],X_rec[x,0]],[X_norm[x,1],X_rec[x,1]],'k--')
    # 输入第一项全是X坐标，第二项都是Y坐标
plt.legend()
plt.show()