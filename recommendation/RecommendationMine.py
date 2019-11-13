# -*- coding:utf-8 -*-
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt


def load_movies():
    """
    读取电影评论
    :return:
    """
    mat = loadmat('./data/ex8_movies.mat')
    Y, R = mat['Y'], mat['R']
    num_movies, num_users = Y.shape
    return Y, R, num_movies, num_users


def load_params():
    """
    读取电影参数
    :return:
    """
    mat = loadmat('./data/ex8_movieParams.mat')
    X = mat['X']
    Theta = mat['Theta']
    num_users = int(mat['num_users'])
    num_movies = int(mat['num_movies'])
    num_features = int(mat['num_features'])
    return X, Theta, num_movies, num_users, num_features


def load_movies_idx():
    """
    读取电影列表
    :return:
    """
    movies = []
    with open('data/movie_ids.txt', 'r') as f:
        for line in f:
            movies.append(''.join(line.strip().split(' ')[1:]))
    return movies


def serialize(X, Theta):
    """
    序列化参数
    :param X:用户参数向量
    :param Theta:电影特征向量
    :return:
    """
    return np.concatenate((X, Theta))


def deserialize(params, num_movies, num_users, num_features):
    """
    反序列化
    :param params: 参数
    :param num_movies: 电影数量
    :param num_users: 用户数量
    :param num_features: 特征数量
    :return:
    """
    return params[:num_movies * num_features].reshape(num_movies, num_features), params[
                                                                                 num_movies * num_features:].reshape(
        num_users, num_features)


def cofiCostFunc(params, Y, R, num_movies, num_users, num_features, l=0.0):
    """

    :param params:
    :param Y:
    :param R:
    :param num_movies:
    :param num_users:
    :param num_features:
    :param l:
    :return:
    """
    X, Theta = deserialize(params, num_movies, num_users, num_features)
    error = 0.5 * np.square((X @ Theta.T * Y) * R).sum()
    reg_1 = 0.5 * np.square(Theta).sum()
    reg_2 = 0.5 * np.square(X).sum()
    return error + reg_1 + reg_2


def cofiGradient(params, Y, R, num_movies, num_users, num_features, l=0.0):
    """
    梯度下降
    :param params:
    :param Y:
    :param R:
    :param num_movies:
    :param num_users:
    :param num_features:
    :param l:
    :return:
    """
    X, Theta = deserialize(params, num_movies, num_users, num_features)
    X_grad = ((X @ Theta.T - Y) * R) @ Theta + 1 * X
    Theta_grad = ((X @ Theta.T - Y) * R).T @ X + 1 * Theta
    return serialize(X_grad, Theta_grad)


def normallizeRatings(Y, R):
    """
    求均值
    :param Y:
    :param R:
    :return:
    """
    return (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1, 1)


if __name__ == '__main__':
    Y, R, _, _ = load_movies()
    # X, Theta, _, _, _ = load_params()
    movies_idx = load_movies_idx()
    # 模拟用户
    my_ratings = np.zeros((1682, 1))
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5
    Y = np.concatenate((Y, my_ratings), axis=1)
    R = np.concatenate((R, my_ratings != 0), axis=1)
    num_movies, num_users = Y.shape
    num_features = 10
    X = np.random.random((num_movies, num_features))
    Theta = np.random.random((num_users, num_features))
    params = serialize(X, Theta)
    Ymean = normallizeRatings(Y, R)
    l = 10
    res = opt.minimize(fun=cofiCostFunc, x0=params, args=(Y, R, num_movies, num_users, num_features, l),
                       jac=cofiGradient, method='TNC', options={'maxiter': 100})
    ret = res.x
    fit_X, fit_Theta = deserialize(ret, num_movies, num_users, num_features)
    pred_mat = fit_X @ fit_Theta.T
    pred = pred_mat[:, -1] + Ymean.flatten()
    pred_sorted_idx = np.argsort(pred)[::-1]
    print("Top recommendations for you:")
    for i in range(10):
        print('Predicting rating %0.1f for movie %s.' \
              % (pred[pred_sorted_idx[i]], movies_idx[pred_sorted_idx[i]]))
