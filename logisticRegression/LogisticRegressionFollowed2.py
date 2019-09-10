# -*- coding:utf-8 -*-
import pandas
import numpy
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report


power = 6
l = 1

data = pandas.read_csv('./data/ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]
fig, ax = plt.subplots()
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')


# plt.show()


def feature_mapping(x1, x2, power):
    """
    特征映射，将特征映射到高阶
    :param x1:
    :param x2:
    :param power:
    :return:
    """
    data = {}
    for i in numpy.arange(power + 1):
        for p in numpy.arange(i + 1):
            data["f{}{}".format(i - p, p)] = numpy.power(x1, i - p) * numpy.power(x2, p)
    return pandas.DataFrame(data)


x1 = data['Test 1'].as_matrix()
x2 = data['Test 2'].as_matrix()
# 获得数据的高阶映射
data1 = feature_mapping(x1, x2, power=power)

# 这里因为做特征映射的时候已经添加了偏置项，所以不用手动添加了。
X = data1.as_matrix()
y = data['Accepted'].as_matrix()
# 初始化theta值
theta = numpy.zeros(X.shape[1])


# ((118, 28), (118,), (28,))


def sigmoid(z):
    """
    定义sigmoid函数
    :param z: 原函数
    :return: 变换后的函数
    """
    return 1.0 / (1 + numpy.exp(-z))


def computeCost(theta, X, y):
    """
    计算代价
    :param X: 输入数据
    :param y:目标数据
    :param theta: theta
    :return:计算后的代价
    """
    # X.T@X等价于X.T.dot(X)
    return numpy.mean(-y * numpy.log(sigmoid(X @ theta)) - (1 - y) * numpy.log(1 - sigmoid(X @ theta)))


def costReg(theta, X, y, l=l):
    # 不惩罚第一项
    _theta = theta[1:]
    reg = (l / (2 * len(X))) * (_theta @ _theta)  # _theta@_theta == inner product
    return computeCost(theta, X, y) + reg


cost = costReg(theta, X, y, l=1)  # 0.69314718056


def gradientDescent(theta, X, y):
    """
    梯度下降计算
    :param X: 输入向量
    :param y: 目标向量
    :param theta: theta
    :return: theta
    """
    for i in range(10000):
        theta = theta - 0.01 * (X.T @ (sigmoid(X @ theta) - y)) / len(X)
    return theta


def gradient(theta, X, y):
    #     '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


def gradientReg(theta, X, y, l=1):
    """
    正则化梯度下降
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    reg = (1 / len(X)) * theta
    reg[0] = 0
    return gradient(theta, X, y) + reg

print(costReg(theta,X,y)) #0.69314718056
print(theta)
print(X)
print(y)
theta = gradientReg(theta, X, y)
print(theta)
"""
[  8.47457627e-03   7.77711864e-05   3.76648474e-02   2.34764889e-02
   3.93028171e-02   3.10079849e-02   3.87936363e-02   1.87880932e-02
   1.15013308e-02   8.19244468e-03   3.09593720e-03   4.47629067e-03
   1.37646175e-03   5.03446395e-02   7.32393391e-03   1.28600503e-02
   5.83822078e-03   7.26504316e-03   1.83559872e-02   2.23923907e-03
   3.38643902e-03   4.08503006e-04   3.93486234e-02   4.32983232e-03
   6.31570797e-03   1.99707467e-02   1.09740238e-03   3.10312442e-02]
"""



print('init cost = {}'.format(costReg(theta, X, y)))
res = opt.minimize(fun=costReg, x0=theta, args=(X, y), method='Newton-CG', jac=gradientReg)
print(res)


def predict(x, theta):
    """
    预测函数
    :param x:
    :param theta:
    :return:
    """
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)


final_theta = res.x
y_pred = predict(X, final_theta)

print(classification_report(y, y_pred))

positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]
fig, ax = plt.subplots()
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
ax.set_title('power = ' + str(power) + '     lambda = '+str(l))
x = numpy.linspace(-1, 1.5, 250)
# 生成网格坐标矩阵
xx, yy = numpy.meshgrid(x, x)
z = feature_mapping(xx.ravel(), yy.ravel(), power).as_matrix()
z = z @ final_theta
z = z.reshape(xx.shape)
plt.contour(xx, yy, z, 0)
plt.ylim(-.8, 1.2)

plt.show()
print()
