#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2020/4/21 19:07
# @Author   : Daishijun
# @Site     : 
# @File     : read_gearbox_data.py
# @Software : PyCharm

import os
import sys
sys.path.append("../models")
sys.path.append("../base")
filename = os.path.basename(__file__)
import numpy as np
import pandas as pd
# import keras
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler




###########################
#        参数设定         #
##########################

csv_dir = '../dataset/WT_Gearbox'
sub_dir = '../saver/WTGear'
ps = 'mm'

###########################
#       预处理函数        #
##########################

def preprocess(train_X=None,test_X=None,preprocessing='st'):
    if preprocessing=='st': # Standardization（标准化）
        prep = StandardScaler() # 标准化
    if preprocessing=='mm': # MinMaxScaler (归一化)
        prep = MinMaxScaler() # 归一化
    train_X = prep.fit_transform(train_X)
    if test_X is not None:
        test_X = prep.transform(test_X)
    return train_X,test_X,prep


###########################
#        读取与提交        #
###########################


def read_data():
    # train = np.loadtxt(os.path.join(os.path.abspath(csv_dir), 'train[cplt].csv'),  # load 训练集样本
    #                    dtype=np.float32, delimiter=',')
    # test = np.loadtxt(os.path.join(os.path.abspath(csv_dir), 'test[cplt].csv'),  # load 训练集标签
    #                   dtype=np.int32, delimiter=',')

    dftrain = pd.read_csv(os.path.join(os.path.abspath(csv_dir), 'gearboxtrainset.csv'))
    dftest = pd.read_csv(os.path.join(os.path.abspath(csv_dir), 'test_20004015-2016-10.csv'))

    print('wrong dftrain rows:{}'.format(dftrain.shape))

    dftrain = dftrain.astype(str)


    dftrain = dftrain[~dftrain['iTempGearBearNDE_1sec'].str.contains('iTempGearBearNDE_1sec')]
    print('changed dftrain rows:{}'.format(dftrain.shape))
    dftrain = dftrain.astype(np.float32)

    train = dftrain.values
    test = dftest.values[:,:-2]

    print(train[:,0])

    print('train shape {}'.format(train.shape))
    print('test shape {}'.format(test.shape))

    train_X = np.array(train[:, :], dtype=np.float32)
    train_Y = np.array(train[:, :], dtype=np.float32)
    test_X = np.array(test, dtype=np.float32)

    train_X, test_X, scaler = preprocess(train_X, test_X)

    print('train_X shape {}'.format(train_X.shape))
    print('train_Y shape {}'.format(train_Y.shape))

    return [train_X, train_Y, test_X], scaler


def submission(predict):
    df = pd.read_csv(os.path.join(os.path.abspath(csv_dir), 'test_20004015-2016-10.csv'))

    print('predict type : {}'.format(type(predict)))
    print('predict shape :{}'.format(predict.shape))
    print('origin df shape : {}'.format(df.shape))

    df.iloc[:,:-2] = predict
    if not os.path.exists(os.path.join(os.path.abspath(sub_dir))): os.makedirs(os.path.join(os.path.abspath(sub_dir)))
    df.to_csv(os.path.join(os.path.abspath(sub_dir), 'sub01[gear].csv'), index=False)  # save 测试集标签
    print('submission file:{}'.format(os.path.join(os.path.abspath(sub_dir), 'sub01[gear].csv')))


if __name__ == "__main__":
    read_data()