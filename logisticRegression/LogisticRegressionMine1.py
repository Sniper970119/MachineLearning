# -*- coding:utf-8 -*-
import numpy
import pandas
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report

data = pandas.read_csv('data/ex2data1.txt', names=['exam1', 'exam2', 'admitted'])

# 数据初始化
# data = (data - data.mean()) / data.std()

fig, ax = plt.subplots()
positive = data[data.admitted.isin(['1'])]
negative = data[data.admitted.isin(['0'])]
data.insert(0, 'Ones', 1)
X = data.iloc[:, 0:3].values
"""
这里要使用-1 ，这样y为向量，而不是矩阵
"""
y = data.iloc[:, -1].values
theta = numpy.zeros(3)

# 数据预览
ax.scatter(positive['exam1'], positive['exam2'], c='r', label='Admitted')
ax.scatter(negative['exam1'], negative['exam2'], c='b', label='Not Admitted')
ax.set_title('data')
ax.set_xlabel('exam1')
ax.set_ylabel('exam2')
ax.legend(loc=1)


# plt.show()


# 相关函数编写
def sigmoid(X):
    """
    sigmoid函数
    :param X:
    :return:
    """
    return 1.0 / (1.0 + numpy.exp(-X))


def cost(theta, X, y):
    """
    计算代价函数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    temp = y * numpy.log(sigmoid(X @ theta)) + (1 - y) * numpy.log(1 - sigmoid(X @ theta))
    return numpy.mean(-temp)


def gradient(theta, X, y):
    """
    单次下降函数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    return (1 / len(X)) * (sigmoid(X @ theta) - y) @ X


def regularized_cost(theta, X, y, l=1):
    """
    正则化代价函数
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    _theta = theta[1:]
    # 计算惩罚项
    reg = (l / (2 * len(X)) * (_theta @ _theta))
    return cost(theta, X, y) + reg


def regularized_gradient(theta, X, y, l=1):
    """
    正则化下降函数
    :return:
    """
    reg = (1 / len(X)) * theta
    reg[0] = 0
    return gradient(theta, X, y) + reg


def predict(X, theta):
    """
    预测函数
    :param x:
    :param theta:
    :return:
    """
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)


# # 计算代价
costValue = cost(theta, X, y)
# print(costValue)

# 计算代价
result = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
print(result)

# 进行预测
final_theta = result.x
print(final_theta)
predict_result = predict(X, final_theta)
# print(classification_report(y, predict_result))

# 绘制图形
coef = -(final_theta / final_theta[2])
x = numpy.linspace(data['exam1'].min(), data['exam1'].max(), 100)
y = x * coef[1] + coef[0]
ax.plot(x, y)
plt.show()
