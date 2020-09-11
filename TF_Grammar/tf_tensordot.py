#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_tensordot.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/11 上午11:03
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf


if __name__ == "__main__":



    # --------------------axes is scalar----------------
    a_0 = tf.ones(shape=[2, 2, 3])
    b_0 = tf.ones(shape=[3, 2, 6])
    # a_axis dimension=> [1, 2]
    # b_axis dimension=> [0, 1]
    c_0 = tf.tensordot(a_0, b_0, axes=2)  # (2, 2, 6)

    # -------------axes is tensor------------------------
    a_1 = tf.ones(shape=[2, 3, 6])
    b_1 = tf.ones(shape=[6, 2])

    a_1_axis = len(a_1.shape)-1  # a_1_axis dimension =>[2]
    b_1_axis = 0 # # b_1_axis dimension =>[0, 1]
    #  sum over the last N axes of a and the first N axes of b in order
    c_1 = tf.tensordot(a_1, b_1, axes=[a_1_axis, b_1_axis])  # (2, 3, 2)

    with tf.Session() as sess:
        print('axes is scalar')
        print(a_0.eval())
        print(b_0.eval())
        print(c_0.eval())
        print('axes is tensor')
        print(a_1.eval())
        print(b_1.eval())
        print(c_1.eval())
