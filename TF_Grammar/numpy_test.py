#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : numpy_test.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/17 下午3:35
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np

if __name__ == "__main__":

    #---------------------stack test---------------------------
    a = [[2] * 3]
    b = np.array(a).transpose()
    print(b.shape)
    c = np.random.rand(3, 2)
    print(c.shape)
    d = np.hstack((b, c))
    print(d)
    print('+' * 40)
    #---------------------inverse(~) test----------------------

    print(sum(~np.array([True, True, True, False]).astype(np.bool)))

    #----------------------argsort test------------------------
    print('+' * 40)
    g = np.array([2, 4, 6, 3])
    # get the indices number array that sorted by axis
    # step 1 get indices map
    # step 2 adjust indices oder by value

    g_s = np.argsort(g)
    print(g_s)
    print(g[g_s[0]])  # 2
    print(g[g_s[1]])  # 3
    print(g[g_s[2]])  # 4
    print(g[g_s[-1]]) # 6

    # ----------------------cumsum test------------------------
    h = np.array([1, 2, 0, 3, 4, 5])
    # the cumulative sum of the elements along a given axis
    # step: [1, 1+2, 1+2+0, 1+2+0+3, 1+2+0+3+4, 1+2+0+3+4+5]
    print(np.cumsum(h)) # [ 1  3  3  6 10 15]

    #-----------------------finfo------------------------------
    # get machine limits for floating point type
    type_limit = np.finfo(np.float64)
    print(type_limit)
    # get smallest representable positive number of the type
    print(type_limit.eps)
