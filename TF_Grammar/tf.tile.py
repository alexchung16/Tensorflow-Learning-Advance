#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf.tile.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/18 下午3:50
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf

if __name__ == "__main__":

    size = 10
    t = tf.constant([[0, 0], [1, 2], [3, 4]], dtype=tf.int32)

    # Pads a tensor
    r_0 = tf.tile(input=t, multiples=[2, 1])

    r = tf.range(size, dtype=tf.int32)
    r_extend_0 = r[:, tf.newaxis]  # (10, 1)
    r_extend_1 = r[tf.newaxis, :]  # (1, 10)

    # get tensor by tail given tensor
    r_extend_0 = tf.tile(r_extend_0, [1, size])  # (10, 10)
    r_extend_1 = tf.tile(r_extend_1, [size, 1]) # (10, 10)


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # print(r1)
        print(r_0.eval())

        print(r.shape)
        print(r_extend_0.eval())
        print(r_extend_1.eval())