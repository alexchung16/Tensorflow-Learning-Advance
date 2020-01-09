#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_greater.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/19 上午9:57
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf


if __name__ == "__main__":

    d = tf.constant([2, 3, 5, 1, 6, 7], dtype=tf.int32)

    g = tf.greater(d, 4)

    m = d * tf.cast(g, dtype=tf.int32)

    with tf.Session() as sess:
        print(sess.run(g))
        print(sess.run(m))

    # [False False  True False  True  True]
    # [0 0 5 0 6 7]