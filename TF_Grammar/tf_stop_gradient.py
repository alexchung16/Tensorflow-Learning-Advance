#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_stop_gradient.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/18 AM 11:26
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [3,2])
    y = tf.placeholder(tf.float32, [3,4])
    w1 = tf.Variable(tf.ones([2,3]))
    w2 = tf.Variable(tf.ones([3,4]))

    hidden = tf.stop_gradient(tf.matmul(x, w1))
    output = tf.matmul(hidden, w2)

    loss = output - y

    optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print("---Before Gradient Descent---")
      print("w1:\n", w1.eval(), "\nw2:\n", w2.eval())
      w1_, w2_, _ = sess.run([w1, w2, optimizer],
                           feed_dict={x:np.random.normal(size = (3,2)),
                                       y:np.random.normal(size = (3,4))})
      print("---After Gradient Descent---")
      print("w1:\n", w1_, "\nw2:\n", w2_)