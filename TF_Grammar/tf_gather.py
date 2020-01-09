#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File tf_gather.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 16/12/2019 PM 20:27

import tensorflow as tf

if __name__ == "__main__":

    a = tf.Variable(initial_value=[0, 1, 2 , 3, 4, 5, 6, 7, 8, 9], dtype=tf.float32)
    b = tf.Variable(initial_value=tf.random_normal(shape=(6, 4)))
    indices = tf.Variable(initial_value=[0, 2, 4])

    # gather slices
    a_g = tf.gather(params=a, indices=indices)
    b_g = tf.gather(params=b, indices=indices)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run([a, a_g]))
        print(sess.run([b, b_g]))
