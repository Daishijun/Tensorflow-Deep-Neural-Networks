#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2020/4/24 21:27
# @Author   : Daishijun
# @Site     : 
# @File     : calupper.py
# @Software : PyCharm

import numpy as np
import pandas as pd
import math
'''根据极值分布的临界值公式，已知尺度，位置和形状参数，计算临界值。'''
def calculatelimit(a, b, s, p=0.01):
    '''
    :param a:     尺度参数
    :param b:     位置
    :param s:     形状
    :param p:     越界概率(重现期的倒数)
    :return:
    '''
    inner = 1- math.pow(-math.log(1-float(p)), -float(s))
    reth = float(b) - float(a)/float(s) * inner
    return reth



if __name__ == '__main__':
    # print(calculatelimit(a=12.76, b=49.85, s=0.323, p=0.01))
    df = pd.read_csv(r'D:\Users\DSJ\MATLABproject\gearalltrained_parmhats_test14_72h.csv', header=None)
    parms_matrix = df.values
    i = 1
    relist = []
    for parms in parms_matrix:
        print(i)
        shape, scale, location = parms
        rth = calculatelimit(scale, location, shape)
        relist.append(rth)
        i +=1
    relist = pd.Series(relist)
    relist.to_csv(r'..\testdata\Gearbox\gearalltrainedReth_test14_72h.csv', index=False, header=None)
    print('ok')