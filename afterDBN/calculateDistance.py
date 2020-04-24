#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2020/4/24 21:20
# @Author   : Daishijun
# @Site     : 
# @File     : calculateDistance.py
# @Software : PyCharm

'''
根据DAE输出的预测序列和真实序列，计算出Re值。
'''
import numpy as np
import pandas as pd

df1 = pd.read_csv(r'D:\Users\DSJ\Pythonproject\github_DBN\Tensorflow-Deep-Neural-Networks\dataset\WT_Gearbox\test_20004015-2016-10.csv', header=None)

df2 = pd.read_csv(r'D:\Users\DSJ\Pythonproject\github_DBN\Tensorflow-Deep-Neural-Networks\saver\WTGear\sub01[gear].csv', header=None)
# df2 = pd.read_csv(r'D:\Users\DSJ\Pythonproject\github_DBN\saved_models\2_26(server)\predict_Y.csv', header=None)

# print(type(df1.values))
# print(df1.values.shape)
target_ = df1.values[:,:-2]
predict_ = df2.values[:,:-2]

#求欧式距离，输入的数组：n_samples * n_features; 输出的数组：1 * n_samples
# print(np.sqrt(np.sum(np.square(target_ - predict_), axis=1)))

distarray = np.sqrt(np.sum(np.square(target_ - predict_), axis=1))

print('target shape:',target_.shape)
print('dist shape:',distarray.shape)


np.savetxt(r'D:\Users\DSJ\Pythonproject\github_DBN\Tensorflow-Deep-Neural-Networks\testdata\Gearbox\dist_gear_sep20.csv', distarray, delimiter=',')