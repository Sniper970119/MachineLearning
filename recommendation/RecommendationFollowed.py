# -*- coding:utf-8 -*-

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt

mat = loadmat('./data/ex8_movies.mat')
# dict_keys(['__header__', '__version__', '__globals__', 'Y', 'R'])
print(mat.keys())
# Y是不同电泳的不同用户评分，R代表用户i对电影j有是否有评分
Y, R = mat['Y'], mat['R']
movies_number, user_number = Y.shape
# (1682, 943) (1682, 943)
print(Y.shape, R.shape)
# # 计算一下用户0 的平均打分成绩  3.8783185840707963
# print(Y[0].sum() / R[0].sum())
#
# 显示一下Y矩阵的图示
fig = plt.figure(figsize=(8, 8 * (1682. / 943.)))
plt.imshow(Y, cmap='rainbow')
plt.colorbar()
plt.ylabel('Movies (%d)' % movies_number, fontsize=20)
plt.ylabel('USers (%d)' % user_number, fontsize=20)
# plt.show()


# 协同过滤
mat = loadmat('./data/ex8_movieParams.mat')
# dict_keys(['__header__', '__version__', '__globals__', 'X', 'Theta', 'num_users', 'num_movies', 'num_features'])
print(mat.keys())

X = mat['X']
Theta = mat['Theta']
num_users = int(mat['num_users'])
num_movies = int(mat['num_movies'])
num_features = int(mat['num_features'])
# 943 1682 10
print(num_users, num_movies, num_features)
# (1682, 10) (943, 10)
print(X.shape, Theta.shape)
# 降低一下数据的大小，目的是让程序运行的更快
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]
# (5, 3) (4, 3)
print(X.shape, Theta.shape)


# 协同过滤代价函数
def serialize(X, Theta):
    """
    展开参数
    :param X:每个电影的打分
    :param Theta: 每个用户的专有电影参数？是这么说的么
    :return:
    """
    #
    return np.r_[X.flatten(), Theta.flatten()]


def deserialize(seq, num_movies, num_users, num_features):
    """
    提取参数
    :param seq:
    :param num_movies: 电影数量
    :param num_users: 用户数量
    :param num_features: 特征数量
    :return:
    """
    return seq[:num_movies * num_features].reshape(num_movies, num_features), seq[num_movies * num_features:].reshape(
        num_users, num_features)


def cofiCostFunc(params, Y, R, num_movies, num_users, num_features, l=0.0):
    """
    代价函数
    :param params: 一维后的参数（X，theta）
    :param Y: 评分矩阵
    :param R: 有无评分矩阵
    :param num_movies: 电影数量
    :param num_users: 用户数量
    :param num_features: 特征数量
    :param l: 正则化参数
    :return:
    """
    X, Theta = deserialize(params, num_movies, num_users, num_features)
    # 计算
    error = 0.5 * np.square((X @ Theta.T - Y) * R).sum()
    # 对电影评分的正则化
    reg1 = 0.5 * l * np.square(Theta).sum()
    # 对电影参数的正则化
    reg2 = 0.5 * l * np.square(X).sum()
    return error + reg1 + reg2


# 22.224603725685675
print(cofiCostFunc(serialize(X, Theta), Y, R, num_movies, num_users, num_features))
# 31.34405624427422
print(cofiCostFunc(serialize(X, Theta), Y, R, num_movies, num_users, num_features, 1.5))


# 下降
def cofiGradient(params, Y, R, num_movies, num_users, num_features, l=0.0):
    """
    计算X和Theta的梯度，并序列化输出
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


# 计算电影评分
# 读取电影列表
movies = []
with open('data/movie_ids.txt', 'r') as f:
    for line in f:
        movies.append(''.join(line.strip().split(' ')[1:]))
# 模拟一个用户的评分
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

for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(my_ratings[i], movies[i])
"""
[4.] ToyStory(1995)
[3.] TwelveMonkeys(1995)
[5.] UsualSuspects,The(1995)
[4.] Outbreak(1995)
[5.] ShawshankRedemption,The(1994)
[3.] WhileYouWereSleeping(1995)
[5.] ForrestGump(1994)
[2.] SilenceoftheLambs,The(1991)
[4.] Alien(1979)
[5.] DieHard2(1990)
[5.] Sphere(1998)
"""
# 读取电影打分矩阵以及用户打分矩阵
mat = loadmat('data/ex8_movies.mat')
Y, R = mat['Y'], mat['R']
# (1682, 943) (1682, 943)
print(Y.shape, R.shape)

# 将新用户的打分添加进去
Y = np.concatenate((Y, my_ratings), axis=1)
R = np.concatenate((R, my_ratings != 0), axis=1)
num_movies, num_users = Y.shape
# 1682 944
print(num_movies, num_users)
# 设定特征向量为10个
num_features = 10


def normallizeRatings(Y, R):
    """
    计算打过分的电影的平均值
    :param Y:
    :param R:
    :return:
    """
    Ymean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1, 1)
    Ynorm = (Y - Ymean) * R
    return Ynorm, Ymean


Ynorm, Ymean = normallizeRatings(Y, R)
# (1682, 944) (1682, 1)
print(Ynorm.shape, Ymean.shape)

X = np.random.random((num_movies, num_features))
Theta = np.random.random((num_users, num_features))
params = serialize(X, Theta)
l = 10

res = opt.minimize(fun=cofiCostFunc, x0=params, args=(Y, R, num_movies, num_users, num_features, l), method='TNC',
                   jac=cofiGradient, options={'maxiter': 100})
ret = res.x
# [0.95054705 0.35474199 0.67897876 ... 0.23782728 0.76704132 0.79748087]
print(ret)
fit_X, fit_Theta = deserialize(ret, num_movies, num_users, num_features)
# 计算预测分数
pred_mat = fit_X @ fit_Theta.T
# 预测分值+平均值
pred = pred_mat[:, -1] + Ymean.flatten()
pred_sorted_idx = np.argsort(pred)[::-1]
print("Top recommendations for you:")
for i in range(10):
    print('Predicting rating %0.1f for movie %s.' \
          % (pred[pred_sorted_idx[i]], movies[pred_sorted_idx[i]]))
"""
Top recommendations for you:
Predicting rating 9.8 for movie MayaLin:AStrongClearVision(1994).
Predicting rating 9.7 for movie Schindler'sList(1993).
Predicting rating 9.6 for movie PatherPanchali(1955).
Predicting rating 9.3 for movie KasparHauser(1993).
Predicting rating 9.3 for movie Faust(1994).
Predicting rating 9.2 for movie Trainspotting(1996).
Predicting rating 9.2 for movie SomeFolksCallItaSlingBlade(1993).
Predicting rating 9.2 for movie Prefontaine(1997).
Predicting rating 9.2 for movie BeautifulThing(1996).
Predicting rating 9.1 for movie GreatDayinHarlem,A(1994).
"""

print("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for movie %s.' % (my_ratings[i], movies[i]))
"""
Original ratings provided:
Rated 4 for movie ToyStory(1995).
Rated 3 for movie TwelveMonkeys(1995).
Rated 5 for movie UsualSuspects,The(1995).
Rated 4 for movie Outbreak(1995).
Rated 5 for movie ShawshankRedemption,The(1994).
Rated 3 for movie WhileYouWereSleeping(1995).
Rated 5 for movie ForrestGump(1994).
Rated 2 for movie SilenceoftheLambs,The(1991).
Rated 4 for movie Alien(1979).
Rated 5 for movie DieHard2(1990).
Rated 5 for movie Sphere(1998).
"""
