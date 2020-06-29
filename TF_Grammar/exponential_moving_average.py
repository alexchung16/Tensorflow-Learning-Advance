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


def ema_comp_ema_corr():
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
        v_t = beta * v_pre + (1-beta) * t
        v_ema.append(v_t)
        v_pre = v_t
    print("v_mea:", v_ema)

    # step 3 correct the ema results
    v_ema_corr = []
    for i, t in enumerate(v_ema):
        v_ema_corr.append(t/(1-np.power(beta, i+1)))
    print("v_ema_corr", v_ema_corr)

    # step 4 plot ema and correction ema reslut
    plt.plot(x_index, raw_tmp, label='raw_tmp')  # Plot some data on the (implicit) axes.
    plt.plot(x_index, v_ema, label='v_ema')  # etc.
    plt.plot(x_index, v_ema_corr, label='v_ema_corr')
    plt.xlabel('time')
    plt.ylabel('T')
    plt.title("exponential moving average")
    plt.legend()
    plt.savefig('./ema.png')
    plt.show()


def ema_corr(data, beta):
    v_ema = []
    v_ema_corr = []
    v_pre = 0
    for i, t in enumerate(data):
        v_t = beta * v_pre + (1 - beta) * t
        v_ema.append(v_t)
        v_pre = v_t

    for i, t in enumerate(v_ema):
        v_ema_corr.append(t / (1 - np.power(beta, i + 1)))
    return v_ema_corr


def ema_beta():

    beta_0 = 0.5
    beta_1 = 0.9
    beta_2 = 0.98
    num_samples = 100

    # step 1 generate random seed
    np.random.seed(0)
    raw_tmp = np.random.randint(30, 38, size=num_samples)
    x_index = np.arange(num_samples)
    # raw_tmp = [35, 34, 37, 36, 35, 38, 37, 37, 39, 38, 37]  # temperature
    print(raw_tmp)

    # step 2 calculate ema result
    v_ema_5 = ema_corr(raw_tmp, beta_0)
    print("v_ema_5", v_ema_5)
    v_ema_9 = ema_corr(raw_tmp, beta_1)
    print("v_ema_9", v_ema_9)
    v_ema_98 = ema_corr(raw_tmp, beta_2)
    print("v_ema_99", v_ema_98)

    # step 4 plot ema and correction ema reslut
    plt.plot(x_index, raw_tmp, label='raw_tmp')  # Plot some data on the (implicit) axes.
    plt.plot(x_index, v_ema_5, label='beta_0.5')  # etc.
    plt.plot(x_index, v_ema_9, label='beta_0.9')
    plt.plot(x_index, v_ema_98, label='beta_0.98')
    plt.xlabel('time')
    plt.ylabel('T')
    plt.title("exponential moving average")
    plt.legend()
    plt.savefig('./ema.png')
    plt.show()


if __name__ == "__main__":
    ema_beta()