# -*- coding:utf-8 -*-
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
from sklearn.metrics import classification_report

"""
相关方法
"""


def load_data(path):
    """
    读取数据
    :param path:
    :param transpose:
    :return:
    """
    data = sio.loadmat(path)
    y = data.get('y')
    # 将y由列向量变成行向量
    y = y.reshape(y.shape[0])
    X = data.get('X')  # 5000*400
    return X, y


def sigmoid(Z):
    """
    sigmoid 函数
    :param Z:
    :return:
    """
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(Z):
    """
    sigmoid 的导函数
    :param Z:
    :return:
    """
    # return sigmoid(Z) * (1 - sigmoid(Z))
    return np.multiply(sigmoid(Z), 1 - sigmoid(Z))


def serialize(theta1, theta2):
    """
    扁平化参数
    :param theta1:
    :param theta2:
    :return:
    """
    return np.concatenate((np.ravel(theta1), np.ravel(theta2)))


def deserialize(theta):
    """
    反扁平化参数
    :param theta:
    :return:
    """
    return theta[:25 * 401].reshape(25, 401), theta[25 * 401:].reshape(10, 26)


def feed_forward(theta, X):
    """
    神经网络计算
    :param theta:
    :param X:
    :return:
    """
    # 反扁平化获取当前参数
    t1, t2 = deserialize(theta)
    # 计算隐藏层激活前数据
    hiden_layar_init_data = X @ t1.T
    # 激活隐藏层函数,并增加一列偏置
    hiden_layar_final_data = np.insert(sigmoid(hiden_layar_init_data), 0, np.ones(X.shape[0]), axis=1)

    # 计算输出层激活前数据
    output_layar_init_data = hiden_layar_final_data @ t2.T
    # 激活输出层数据
    output_layar_final_data = sigmoid(output_layar_init_data)
    # 分别返回这四个数据
    return hiden_layar_init_data, hiden_layar_final_data, output_layar_init_data, output_layar_final_data


def gradient(theta, X, y, l=1):
    """
    梯度下降函数
    :return:
    """
    # 反序列化theta
    t1, t2 = deserialize(theta)
    # 初始化两个个t1,t2一样大的变量，用来存放需要变化的量以进行负反馈
    delta1 = np.zeros(t1.shape)
    delta2 = np.zeros(t2.shape)
    # 获取神经网络计算结果
    hiden_layar_init_data, hiden_layar_final_data, \
    output_layar_init_data, output_layar_final_data = feed_forward(theta, X)
    data_length = X.shape[0]
    for i in range(data_length):
        # 计算输出层误差
        output_layar_final_data_error = output_layar_final_data[i] - y[i]
        # 对隐藏层处理前的数据添加偏置列
        # hiden_layar_init_data_with_bias = np.insert(hiden_layar_init_data[i], 0, 1)
        hiden_layar_init_data_with_bias = np.insert(hiden_layar_init_data[i], 0, np.ones(1))
        # 计算隐藏层的误差
        hiden_layar_final_data_error = (t2.T @ output_layar_final_data_error) \
                                       * sigmoid_derivative(hiden_layar_init_data_with_bias)
        # 对delta进行反馈计算
        # delta1 += output_layar_final_data_error.T @ hiden_layar_final_data[i]
        delta2 += np.matrix(output_layar_final_data_error).T @ np.matrix(hiden_layar_final_data[i])
        delta1 += np.matrix(hiden_layar_final_data_error[1:]).T @ np.matrix(X[i])
    delta1 = delta1 / data_length
    delta2 = delta2 / data_length
    return serialize(delta1, delta2)


def regularized_gradient(theta, X, y, l=1):
    """
    正则化梯度下降函数
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    # 原始参数去扁平化
    t1, t2 = deserialize(theta)
    # 进行下降之后的参数，这里为了防止delta的参数差距过大，需要依据原数据（t1 t2）对参数进行正则化
    delta1, delta2 = deserialize(gradient(theta, X, y))
    # 数据个数
    data_length = X.shape[0]
    # 不惩罚第一项 所以t1、t2的第一项设为0
    t1[:, 0] = 0
    t2[:, 0] = 0
    reg_t1 = (l / data_length) * t1
    reg_t2 = (l / data_length) * t2
    delta1 = delta1 + reg_t1
    delta2 = delta2 + reg_t2
    delta1_sum = np.sum(delta1)
    delta2_sum = np.sum(delta2)
    a2 = np.sum(serialize(delta1, delta2))
    return serialize(delta1, delta2)


def cost(theta, X, y, l=1):
    """
    代价计算函数
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    data_length = X.shape[0]
    _, _, _, output_layar_final_data = feed_forward(theta, X)
    pair_computation = -np.multiply(y, np.log(output_layar_final_data)) - np.multiply((1 - y), np.log(
        1 - output_layar_final_data))
    return pair_computation.sum() / data_length
    # return np.sum(
    #     -(y * np.log(output_layar_final_data)) - ((1 - y) * np.log(1 - output_layar_final_data))) / data_length


def expand_y(y):
    """
    处理结果向量
    将1 2 这种结果转化为 0 1 0 0 0 0 0  这种向量
    :param y:
    :return:
    """
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1
        res.append(y_array)
    return np.array(res)


def regularized_cost(theta, X, y, l=1):
    """
    正则化代价计算函数
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    t1, t2 = deserialize(theta)
    data_length = X.shape[0]
    # 不惩罚第一项，将第一列去掉
    reg_t1 = (l / (2 * data_length)) * np.power(t1[:, 1:], 2).sum()
    reg_t2 = (l / (2 * data_length)) * np.power(t2[:, 1:], 2).sum()
    cost1 = cost(theta, X, y)
    a2 = cost(theta, X, y, l) + reg_t1 + reg_t2
    return cost(theta, X, y, l) + reg_t1 + reg_t2


def show_accuracy(theta, X, y):
    """
    输出准确率
    :param theta:
    :param X:
    :param y:
    :return:
    """
    _, _, _, h = feed_forward(theta, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))


def random_init(size):
    """
    随机初始化
    :param size:
    :return:
    """
    return np.random.uniform(-0.12, 0.12, size)


def network_training(X, y):
    """
    神经网络训练
    :param X:
    :param y:
    :return:
    """
    init_theta = random_init(401 * 25 + 26 * 10)
    # init_theta = np.zeros(10285)
    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       jac=regularized_gradient,
                       method='TNC')
    return res


def plot_hidden_layer(theta):
    """
    绘制隐藏层
    theta: (10285, )
    """
    final_theta1, _ = deserialize(theta)
    hidden_layer = final_theta1[:, 1:]  # ger rid of bias term theta

    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5, 5))

    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


if __name__ == '__main__':
    X, y = load_data('./data/ex4data1.mat')
    y_raw = expand_y(y)
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    res = network_training(X, y_raw)
    print(res)
    show_accuracy(res.x, X, y)
    plot_hidden_layer(res.x)
    plt.show()
