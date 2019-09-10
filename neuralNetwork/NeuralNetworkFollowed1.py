# -*-coding:utf-8 -*-
import matplotlib
import numpy
import pandas
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
from sklearn.metrics import classification_report

"""
逻辑回归
"""


def load_data(path, transpose=True):
    """
    文件读取
    :param path:
    :param transpose: 是否转置
    :return:
    """
    data = sio.loadmat(path)
    y = data.get('y')
    # 将y由列向量变成行向量
    y = y.reshape(y.shape[0])
    X = data.get('X')  # 5000*400

    if transpose:
        # 转置当前数据中,可以理解为将图片旋转,将数据分为5000个20*20的矩阵
        X = numpy.array([im.reshape((20, 20)).T for im in X])
        # 将旋转后的图片展开成一行 5000*400
        X = numpy.array([im.reshape(400) for im in X])
    return X, y


X, y = load_data('data/ex3data1.mat')


# (5000, 400)      (5000,)
# print(numpy.shape(X))
# print(numpy.shape(y))


def plot_an_image(image):
    """
    绘图函数
    :param image:
    :return:
    """
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(numpy.array([]))  # just get rid of ticks
    plt.yticks(numpy.array([]))


# 随机绘制一张图片
pick_one = numpy.random.randint(0, 5000)
plot_an_image(X[pick_one, :])
plt.show()
print('this should be {}'.format(y[pick_one]))

def plot_100_image(X):
    """
    #绘图函数，画100张图片
    X :
    """
    # 获得图片大小（根号下400）
    size = int(numpy.sqrt(X.shape[1]))

    # 随机从X中选择100组数据
    sample_idx = numpy.random.choice(numpy.arange(X.shape[0]), 100)
    # 取数随机数据
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(numpy.array([]))
            plt.yticks(numpy.array([]))


# 随机绘制100张图片
plot_100_image(X)
plt.show()


# 方法编写

def sigmoid(z):
    """
    sigmoid
    :param z:
    :return:
    """
    return 1 / (1 + numpy.exp(-z))


def cost(theta, X, y):
    """
    代价函数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    return numpy.mean(-y * numpy.log(sigmoid(X @ theta)) - (1 - y) * numpy.log(1 - sigmoid(X @ theta)))


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
    reg = (l / (2 * len(X)) * (_theta @ _theta))
    return cost(theta, X, y) + reg


def gradient(theta, X, y):
    """
    下降函数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    return (1 / len(X)) * (sigmoid(X @ theta) - y) @ X


def regularized_gradient(theta, X, y, l=1):
    """
    正则化的下降函数
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


def logistic_regression(X, y, l=1):
    """generalized logistic regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = numpy.zeros(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    # get trained parameters
    final_theta = res.x

    return final_theta


# 初始化数据
raw_X, raw_y = load_data('data/ex3data1.mat')
X = numpy.insert(raw_X, 0, values=numpy.ones(raw_X.shape[0]), axis=1)
# 对应行为为该值的图片的索引
y_matrix = []
for k in range(1, 11):
    y_matrix.append((raw_y == k).astype(int))
# 最后一列 k=10为0，把最后一列放到第一列
y_matrix = [y_matrix[-1]] + y_matrix[:-1]
y = numpy.array(y_matrix)
# 初始化theta
theta = numpy.zeros(X.shape[1])
# result = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y[0]), method="TNC", jac=regularized_gradient,
#                       options={'disp': True}).x
# # # print(result)
# y_pred = predict(X, result)
# print('Accuracy={}'.format(numpy.mean(y[0] == y_pred)))
k_theta = numpy.array([logistic_regression(X, y[k]) for k in range(10)])
# print(k_theta.shape)
prob_matrix = sigmoid(X @ k_theta.T)
# 将科学计数法形式转换为小数点，方便输出
numpy.set_printoptions(suppress=True)
print(prob_matrix)
y_pred = numpy.argmax(prob_matrix, axis=1)  # 返回沿轴axis最大值的索引，axis=1代表行
print(y_pred)

y_answer = raw_y.copy()
y_answer[y_answer == 10] = 0

print(classification_report(y_answer, y_pred))
"""
             precision    recall  f1-score   support

          0       0.97      0.99      0.98       500
          1       0.95      0.99      0.97       500
          2       0.95      0.92      0.93       500
          3       0.95      0.91      0.93       500
          4       0.95      0.95      0.95       500
          5       0.92      0.92      0.92       500
          6       0.97      0.98      0.97       500
          7       0.95      0.95      0.95       500
          8       0.93      0.92      0.92       500
          9       0.92      0.92      0.92       500

avg / total       0.94      0.94      0.94      5000
"""

"""
神经网络部分

"""


def load_weight(path):
    """
    读取神经网络权重
    :param path:
    :return:
    """
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


theta1, theta2 = load_weight('data/ex3weights.mat')
# 隐藏层（15+1）个  输出层 10 个
# (25, 401) (10, 26)
print(theta1.shape, theta2.shape)

X, y = load_data('data/ex3data1.mat', transpose=False)

X = numpy.insert(X, 0, values=numpy.ones(X.shape[0]), axis=1)  # intercept

# (5000, 401) (5000,)
print(X.shape, y.shape)

# a1 = X
# 对隐藏层进行计算
z2 = X @ theta1.T  # (5000, 401) @ (25,401).T = (5000, 25)
# (5000, 25)
print(z2.shape)
# 添加隐藏层的0号结点
z2 = numpy.insert(z2, 0, values=numpy.ones(z2.shape[0]), axis=1)
# s激活   (5000, 26)
a2 = sigmoid(z2)
print(a2.shape)
# 对输出层进行计算
z3 = a2 @ theta2.T
print(z3.shape)
# s激活   (5000, 10)
a3 = sigmoid(z3)
print(a3)
# 寻找输出层最大的预测数字
y_pred = numpy.argmax(a3, axis=1) + 1  # 返回沿轴axis最大值的索引，axis=1代表行
# (5000,)
print(y_pred.shape)
"""
             precision    recall  f1-score   support

          1       0.97      0.98      0.97       500
          2       0.98      0.97      0.97       500
          3       0.98      0.96      0.97       500
          4       0.97      0.97      0.97       500
          5       0.98      0.98      0.98       500
          6       0.97      0.99      0.98       500
          7       0.98      0.97      0.97       500
          8       0.98      0.98      0.98       500
          9       0.97      0.96      0.96       500
         10       0.98      0.99      0.99       500

avg / total       0.98      0.98      0.98      5000
"""
print(classification_report(y, y_pred))
print()
