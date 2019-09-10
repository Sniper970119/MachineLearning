# -*-coding:utf-8 -*-
import numpy
import pandas
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report

data = pandas.read_csv('data/ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])

# 数据初始化
positive = data[data.Accepted.isin([1])]
negative = data[data.Accepted.isin([0])]
data.insert(0, 'Ones', 1)

# 数据预览
fig, ax = plt.subplots()
ax.scatter(positive['Test 1'], positive['Test 2'], c='r', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], c='b', label='Not Accepted')
ax.set_xlabel('Test 1')
ax.set_ylabel('Test 2')
ax.set_title('data')


# plt.show()


# 相关函数
def sigmoid(X):
    """
    sigmoid函数
    :param X:
    :return:
    """
    return 1.0 / (1.0 + numpy.exp(-X))


def cost(theta, X, y):
    """
    代价函数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    return numpy.mean(-y * numpy.log(sigmoid(X @ theta)) - (1 - y) * numpy.log(1 - sigmoid(X @ theta)))


def gradient(theta, X, y):
    """
    下降函数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    return (1 / len(X)) * (sigmoid(X @ theta) - y) @ X


def feature_mapping(x1, x2, power):
    """
    高阶映射
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


def regularized_cost(theta, X, y, l=1):
    """
    正则化的代价函数
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    # 惩罚theta 0项不惩罚
    _theta = theta[1:]
    reg = (l / (2 * len(X))) * (_theta @ _theta)
    return cost(theta, X, y) + reg


def regularized_gradient(theta, X, y, l=1):
    """
    正则化下降函数
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    reg = (1 / len(X)) * theta
    reg[0] = 0
    return gradient(theta, X, y) + reg


def predict(x, theta):
    """
    预测函数
    :param x:
    :param theta:
    :return:
    """
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)


# 初始化数据
x1 = data['Test 1'].as_matrix()
x2 = data['Test 2'].as_matrix()
data1 = feature_mapping(x1, x2, 6)
X = data1.as_matrix()
y = data['Accepted'].as_matrix()
# 初始化theta值
theta = numpy.zeros(X.shape[1])
theta = regularized_gradient(theta, X, y)
# print(theta)

# print('init cost = {}'.format(cost(theta, X, y)))
result = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
# print(result)

final_theta = result.x
y_pred = predict(X, final_theta)
# print(classification_report(y, y_pred))

# 绘图
x = numpy.linspace(-1, 1.5, 250)
xx, yy = numpy.meshgrid(x, x)
# .ravel 将多维数组转化为一维数组
z = feature_mapping(xx.ravel(), yy.ravel(), 6).as_matrix()
z = z @ final_theta
z = z.reshape(xx.shape)
plt.contour(xx, yy, z, 0)
# 设置y轴的范围
plt.ylim(-.8, 1.2)
ax.legend(loc=1)
plt.show()
