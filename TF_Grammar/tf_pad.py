#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File tf_pad.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 20/11/2019 PM 20:44

import tensorflow as tf

if __name__ == "__main__":

    r0 = tf.random_uniform(shape=(2, 3))
    r1 = tf.random_uniform(shape=(2, 3, 4))

    pad0 = tf.constant([[1, 2], [2, 1]])

    # Pads a tensor
    result0 = tf.pad(tensor=r0, paddings=pad0, mode="CONSTANT")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        r0, result0 = sess.run([r0, result0])
        print(r0)
        print(result0)




