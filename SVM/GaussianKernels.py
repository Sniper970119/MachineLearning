# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt

from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio


# kernek function
# def gaussian_kernel(x1, x2, sigma):
#     return np.exp(- np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))

#
# x1 = np.array([1, 2, 1])
# x2 = np.array([0, 4, -1])
# sigma = 2
#
# # 0.324652467358
# print(gaussian_kernel(x1, x2, sigma))

mat = sio.loadmat('./data/ex6data2.mat')
print(mat.keys())
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')


sns.lmplot('X1', 'X2', hue='y', data=data,
           height=5,
           fit_reg=False,
           scatter_kws={"s": 10})

plt.show()

"""
线性核函数kernel='linear'
多项式核函数kernel='poly'
径向基核函数kernel='rbf'
sigmod核函数kernel='sigmod'
"""
svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)

print(svc)

svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])


predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob)

plt.show()