#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : AdaGrad
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/7/27 15:56
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# 测试数据
x_data = [338, 333, 328, 207, 226, 25, 179, 60, 208, 606]
y_data = [640, 633, 619, 393, 428, 27, 193, 66, 226, 1591]

# 绘制权重和偏置等高线数据准备
w_y = np.arange(-5, 5, 0.1) # 权重为Y轴
b_x = np.arange(-200, -100, 1) # 偏置为X轴
# 存储各个网格点损失函数值(记录高度值)
Z = np.zeros(shape=(len(b_x), len(w_y)))
# 生成网格阵
b_X, w_Y = np.meshgrid(b_x, w_y)

for i in range(len(b_x)):
    for j in range(len(w_y)):
        b = b_x[i]
        w = w_y[j]
        Z[j][i] = 0.0
        # 计算测试数据在网格点对应权重偏置的损失函数均值: L(f)=（y-(wx+b)）**2
        for n in range(len(x_data)):
            Z[j][i] += (y_data[n] - w*x_data[n]-b)**2
        Z[j][i] /= len(x_data)

# 迭代更新权重和偏置参数
# 初始化权重(weight)
w = -4
# 初始化偏置(bias)
b = -120
# 设置学习率(learning rating)
lr = 1.0

# 设置迭代次数
iteration = 100000

# 初始化权重和偏置学习率
lr_b = 0.0
lr_w = 0.0

# 记录迭代过程权重和偏置
b_iterate = [b]
w_iterate = [w]

# Adagrad 算法能够在训练中自动的对learning rate进行调整，对于出现频率较低参数采用较大的α更新；
# 相反，对于出现频率较高的参数采用较小的α更新。因此，Adagrad非常适合处理稀疏数据。
# 执行参数更新
for i in range(iteration):
    # 初始化权重梯度
    w_gradient = 0.0
    # 初始化偏置梯度
    b_gradient = 0.0

    # 计算测试数据的损失函数梯度之和
    for n in range(len(x_data)):
        w_gradient += -2.0*lr*(y_data[n]-w*x_data[n]-b)*x_data[n]
        b_gradient += -2.0*lr*(y_data[n]-w*x_data[n]-b)*1.0

    # 定制化权重和学习率
    lr_w += w_gradient**2
    lr_b += b_gradient**2
    # 更新权重和偏置
    w -= (lr/np.sqrt(lr_w))*w_gradient
    b -= (lr/np.sqrt(lr_b))*b_gradient

    # 存储迭代权重和偏置
    b_iterate.append(b)
    w_iterate.append(w)

if __name__ == "__main__":

    # 准备画布
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('bias')
    ax.set_ylabel('weight')
    # 绘制等高线图
    plt.contourf(b_X, w_Y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
    # 绘制最优值权重和偏置矩阵点
    # ms和marker分别代表指定点的长度和宽度
    plt.plot([-188.4], [2.67], 'x', ms=6, marker=6, color='orange')
    # 绘制权重和偏置迭代过程路径
    plt.plot(b_iterate, w_iterate, 'o-', ms=3, lw=1.5, color='black')
    plt.xlim(-200, -100)
    plt.ylim(-5, 5)
    plt.title('gradient descent optimize by Adagrad ')
    # plt.xlabel(r'$b$', fontsize=16)
    # plt.ylabel(r'$w$', fontsize=16)
    plt.show()