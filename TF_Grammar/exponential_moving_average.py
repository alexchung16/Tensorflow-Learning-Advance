#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : exponential_moving_average.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/6/29 上午11:49
# @ Software   : PyCharm
#-------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    beta = 0.9
    num_samples = 100

    # step 1 generate random seed
    np.random.seed(0)
    raw_tmp= np.random.randint(32, 38, size=num_samples)
    x_index = np.arange(100)
    # raw_tmp = [35, 34, 37, 36, 35, 38, 37, 37, 39, 38, 37]  # temperature
    print(raw_tmp)

    # step 2 calculate ema result and do not use correction
    v_ema = []
    v_pre = 0
    for i, t in enumerate(raw_tmp):
        v_t = 0.9 * v_pre + 0.1 * t
        v_ema.append(v_t)
        v_pre = v_t
    print(v_ema)

    # step 3 correct the ema results
    v_ema_corr = []
    for i, t in enumerate(v_ema):
        v_ema_corr.append(t/(1-np.power(beta, i+1)))
    print(v_ema_corr)


    # step 4 plot ema and correction ema reslut
    plt.plot(x_index, raw_tmp, label='raw_tmp')  # Plot some data on the (implicit) axes.
    plt.plot(x_index, v_ema, label='v_ema')  # etc.
    plt.plot(x_index, v_ema_corr, label='v_ema_corr')
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("exponential moving average")
    plt.legend()
    plt.savefig('./ema.png')
    plt.show()