# -*- coding:utf-8 -*-
from sklearn import svm

import numpy as np
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('./data/ex6data3.mat')
# print(mat.keys())

training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
training['y'] = mat.get('y')

cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
cv['y'] = mat.get('yval')

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# gamma to comply with sklearn parameter name
# 符合sklearn的参数gamma
combination = [(C, gamma) for C in candidate for gamma in candidate]
# 81
# print(len(combination))

search = []

for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(training[['X1', 'X2']], training['y'])
    search.append(svc.score(cv[['X1', 'X2']], cv['y']))


# 寻找效果最好的一组参数
best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]

# 0.965 (0.3, 100)
# print(best_score, best_param)

