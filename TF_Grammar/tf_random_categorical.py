#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_random_categorical.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/14 下午7:02
# @ Software   : PyCharm
#-------------------------------------------------------


import tensorflow as tf


if __name__ == "__main__":


    logits = tf.log([[99., 1., 1., 1.],
                     [0., 1., 2., 99.]])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.random.set_random_seed(123)
        num_samples = 30
        cat = tf.random.categorical(logits, num_samples)
        print(sess.run(cat))