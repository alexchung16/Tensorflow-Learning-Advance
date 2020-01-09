#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_concat.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/18 下午5:19
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf

if __name__ == "__main__":

    l = []
    a = tf.constant([[2, 3], [4, 5]])
    b = tf.constant([[6, 7], [8, 9]])
    l.append(a)
    l.append(b)
    c = tf.concat(l, axis=0)
    d = tf.concat(l, axis=1)

    with tf.Session() as sess:

        print(sess.run([a, b]))
        print(sess.run([c, d]))