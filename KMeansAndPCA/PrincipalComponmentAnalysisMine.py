# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def loadData():
    """
    读取数据
    :return:
    """
    mat = sio.loadmat('data/ex7data1.mat')
    X = mat['X']
    return X, mat


def featureNormalize(X):
    """
    数据标准化
    :param X:
    :return:
    """
    means = X.means(axis=0)
    stds = X.std(axis=0)
    X_norm = (X - means) / stds
    return X_norm, means, stds


def pca(X):
    """
    pca
    :param X:
    :return:
    """
    # 协方差
    sigma = (X.T * X) / len(X)
    U, S, V = np.linalg.svd(sigma)
    return U, S, V


def projectData(X, U, K):
    """
    数据映射
    :param X:
    :param U:
    :param K:
    :return:
    """
    Z = X @ U[:, :K]
    return Z


def recoverData(Z, U, K):
    """
    恢复数据
    :param Z:
    :param U:
    :param K:
    :return:
    """
    X_rec = Z @ U[:, :K].T
    return X_rec


if __name__ == '__main__':
    pass