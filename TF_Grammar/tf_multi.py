#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File tf_multi.py
# @ Description :
# @ Author alexchung
# @ Time 16/12/2019 PM 15:36

import tensorflow as tf

if __name__ == "__main__":

    #------------------------multiply and tf.newaxis----------------------
    a = tf.constant([0, 0, 2, 3], dtype=tf.float32)
    b = tf.constant([0.5, 1.0, 2.0], shape=(3, 1), dtype=tf.float32)
    b_1 = tf.constant([0.5, 1.0, 2.0], dtype=tf.float32)
    c = a * b
    d = tf.sqrt(b_1)
    # expend dimension
    e = d[:, tf.newaxis]
    f = c / e

    #------------------------tf.range and tf.meshgrid--------------------------------
    m = tf.range(4, dtype=tf.float32) * [2]
    n = tf.range(6, dtype=tf.float32) * [2]

    # broadcasts parameters
    p, q = tf.meshgrid(m, n)

    p1, q1 = tf.meshgrid(b, p)



    with tf.Session() as sess:
        print(sess.run(c))
        print(sess.run(e))
        print(sess.run(f))

        # print(sess.run(m))
        # print(sess.run(n))
        print(sess.run([p, q]))
        print(sess.run(q1))
        print(sess.run(tf.shape(p1)))
        print(sess.run(tf.shape(q1)))

