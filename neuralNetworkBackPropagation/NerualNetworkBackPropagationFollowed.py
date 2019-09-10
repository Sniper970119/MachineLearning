# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report


def load_data(path, transpose=True):
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

    if transpose:
        # 转置当前数据中,可以理解为将图片旋转,将数据分为5000个20*20的矩阵
        X = numpy.array([im.reshape((20, 20)).T for im in X])
        # 将旋转后的图片展开成一行 5000*400
        X = numpy.array([im.reshape(400) for im in X])
    return X, y


X, _ = load_data('./data/ex4data1.mat', transpose=False)


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
# plot_100_image(X)
# plt.show()


"""
前馈神经网络算法
"""
X_raw, y_raw = load_data('./data/ex4data1.mat', transpose=False)
# 增加全部为0的一列  (5000, 401)
X = numpy.insert(X_raw, 0, numpy.ones(X_raw.shape[0]), axis=1)


def expand_y(y):
    """
    处理结果向量
    将1 2 这种结果转化为 0 1 0 0 0 0 0  这种向量
    :param y:
    :return:
    """
    res = []
    for i in y:
        y_array = numpy.zeros(10)
        y_array[i - 1] = 1
        res.append(y_array)

    return numpy.array(res)


# 获得结果集处理后的目标向量
y = expand_y(y_raw)


def load_weight(path):
    """
    读取权重
    :param path:
    :return:
    """
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


t1, t2 = load_weight('./data/ex4weights.mat')


def serialize(a, b):
    """
    当我们使用高级优化方法来优化神经网络时，我们需要将多个参数矩阵展开，才能传入优化函数，然后再恢复形状。
    序列化2矩阵
    这个方法的目的就是使参数扁平化
    # 在这个nn架构中，我们有theta1（25,401），theta2（10,26），它们的梯度是delta1，delta2
    :param a:
    :param b:
    :return:
    """
    # concatenate 方法可以完成数组拼接
    # ravel 方法可以将数据扁平化
    # 这个有个类似方法叫 flatten 该方法也可以做数据扁平化，和raval的区别是ravel返回的是引用，对ravel后的
    # 对象进行修改会影响到原数据，而flatten则会先copy 不会对原数据产生影响。
    return numpy.concatenate((numpy.ravel(a), numpy.ravel(b)))


# 扁平化参数，25*401+10*26=10285
theta = serialize(t1, t2)


def feed_forward(theta, X):
    """apply to architecture 400+1 * 25+1 *10
    X: 5000 * 401
    """

    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]
    a1 = X  # 5000 * 401

    z2 = a1 @ t1.T  # 5000 * 25
    a2 = numpy.insert(sigmoid(z2), 0, numpy.ones(m), axis=1)  # 5000*26

    z3 = a2 @ t2.T  # 5000 * 10
    h = sigmoid(z3)  # 5000*10, this is h_theta(X)

    return a1, z2, a2, z3, h  # you need all those for backprop


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


def deserialize(seq):
    """
    将扁平化处理之后的数据恢复到之前的样子（两个theta
    :param seq:
    :return:
    """
    #     """into ndarray of (25, 401), (10, 26)"""
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


# h是计算后的结果集
_, _, _, _, h = feed_forward(theta, X)


# print(h)


def cost(theta, X, y):
    """
    代价函数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    m = X.shape[0]  # get the data size m

    _, _, _, _, h = feed_forward(theta, X)

    # h是已经激活之后的结果集 所以这里h不需要进行激活
    pair_computation = -numpy.multiply(y, numpy.log(h)) - numpy.multiply((1 - y), numpy.log(1 - h))
    return pair_computation.sum() / m


# cost(theta, X, y)
# 0.287629165161
# print(cost(theta, X, y))

"""
负反馈神经网络
"""


def regularized_cost(theta, X, y, l=1):
    """
    正则化代价函数
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]
    # 忽略第一项，计算t1和t2的代价
    reg_t1 = (l / (2 * m)) * numpy.power(t1[:, 1:], 2).sum()
    reg_t2 = (l / (2 * m)) * numpy.power(t2[:, 1:], 2).sum()

    return cost(theta, X, y) + reg_t1 + reg_t2


# 0.383769859091
# print(regularized_cost(theta, X, y))


def sigmoid_gradient(z):
    """
    sigmoid的导函数，梯度下降时使用
    """
    return numpy.multiply(sigmoid(z), 1 - sigmoid(z))


# 0.25
# print(sigmoid_gradient(0))


def gradient(theta, X, y):
    """
    梯度下降
    :param theta:
    :param X:
    :param y:
    :return:
    """
    # 将扁平化的数据拆分
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]
    # 初始化两个delta
    delta1 = numpy.zeros(t1.shape)  # (25, 401)
    delta2 = numpy.zeros(t2.shape)  # (10, 26)

    # 分别是输入层  隐藏层激活前 隐藏层激活后  输出层激活前 输出层激活后
    a1, z2, a2, z3, h = feed_forward(theta, X)

    # 对每一组数据进行遍历
    for i in range(m):
        a1i = a1[i, :]  # (1, 401)
        z2i = z2[i, :]  # (1, 25)
        a2i = a2[i, :]  # (1, 26)

        hi = h[i, :]  # (1, 10)
        yi = y[i, :]  # (1, 10)

        d3i = hi - yi  # (1, 10)

        z2i = numpy.insert(z2i, 0, numpy.ones(1))  # make it (1, 26) to compute d2i
        d2i = numpy.multiply(t2.T @ d3i, sigmoid_gradient(z2i))  # (1, 26)

        # careful with np vector transpose  注意矩阵转置
        delta2 += numpy.matrix(d3i).T @ numpy.matrix(a2i)  # (1, 10).T @ (1, 26) -> (10, 26)
        delta1 += numpy.matrix(d2i[1:]).T @ numpy.matrix(a1i)  # (1, 25).T @ (1, 401) -> (25, 401)

    delta1 = delta1 / m
    delta2 = delta2 / m

    return serialize(delta1, delta2)


d1, d2 = deserialize(gradient(theta, X, y))


# print(d1, d2)

def gradient_checking(theta, X, y, epsilon, regularized=False):
    """
    梯度校验
    :param theta:
    :param X:
    :param y:
    :param epsilon:
    :param regularized:
    :return:
    """

    def a_numeric_grad(plus, minus, regularized=False):
        """
        calculate a partial gradient with respect to 1 theta
        计算相对于1 θ 的部分梯度
        """
        if regularized:
            return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (epsilon * 2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

    theta_matrix = expand_array(theta)  # expand to (10285, 10285)
    epsilon_matrix = np.identity(len(theta)) * epsilon

    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix

    # calculate numerical gradient with respect to all theta
    # 计算相对于所有theta的数值梯度
    numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
                             for i in range(len(theta))])

    # analytical grad will depend on if you want it to be regularized or not
    # 分析毕业将取决于你是否希望它正规化
    analytic_grad = regularized_gradient(theta, X, y) if regularized else gradient(theta, X, y)

    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # the diff below should be less than 1e-9
    # this is how original matlab code do gradient checking
    # 如果你有正确的实现，并假设你使用EPSILON = 0.0001
    # 下面的差异应小于1e - 9
    # 这是原始matlab代码执行渐变检查的方式
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print(
        'If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(
            diff))


def expand_array(arr):
    """replicate array into matrix
    [1, 2, 3]

    [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]
    """
    # turn matrix back to ndarray
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))


# gradient_checking(theta, X, y, epsilon= 0.0001)#这个运行很慢，谨慎运行


def regularized_gradient(theta, X, y, l=1):
    """don't regularize theta of bias terms"""
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))
    t1, t2 = deserialize(theta)

    t1[:, 0] = 0
    reg_term_d1 = (l / m) * t1
    delta1 = delta1 + reg_term_d1

    t2[:, 0] = 0
    reg_term_d2 = (l / m) * t2
    delta2 = delta2 + reg_term_d2

    return serialize(delta1, delta2)


# gradient_checking(theta, X, y, epsilon=0.0001, regularized=True)#这个运行很慢，谨慎运行


def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


def nn_training(X, y):
    """regularized version
    the architecture is hard coded here... won't generalize
    """
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res


res = nn_training(X, y)  # 慢
print(res)

_, y_answer = load_data('./data/ex4data1.mat')
print(y_answer[:20])

final_theta = res.x


def show_accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)

    y_pred = np.argmax(h, axis=1) + 1

    print(classification_report(y, y_pred))


def plot_hidden_layer(theta):
    """
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


plot_hidden_layer(final_theta)
plt.show()
