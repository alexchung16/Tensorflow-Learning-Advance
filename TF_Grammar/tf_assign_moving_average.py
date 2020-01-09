#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File batch_normalization.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 5/11/2019 PM 16:30


import os
import tensorflow as tf
from tensorflow.python.training import moving_averages

if __name__ == "__main__":

    a = tf.Variable(initial_value=tf.random_uniform(shape=(3, 2), minval=0, maxval=3))
    b = tf.Variable(initial_value=tf.random_uniform(shape=(2, 3), minval=0, maxval=3))

    input = tf.Variable(initial_value=tf.random_uniform(shape=(2, 3, 4), minval=0, maxval=3))
    axis = list(range(len(input.get_shape()) - 1))
    mean, variance = tf.nn.moments(x=input, axes=axis, name='moment')
    params_shape = input.get_shape()[-1:]
    moving_mean = tf.get_variable(name='moving_mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
    update_moving_mean = moving_averages.assign_moving_average(variable=moving_mean, value=mean, decay=0.8)
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)
        print(a.eval())
        print(b.eval())
        # tf.assign
        # tf.assign value to ref
        a_ = tf.assign(ref=a, value=b, validate_shape=False)
        print(a_.eval())

        # assign_moving_average
        mean_0, moving_mean_0 = sess.run([mean, moving_mean])
        print(mean_0)
        print(moving_mean_0)
        mean, moving_mean, update_moving_mean = sess.run([mean, moving_mean, update_moving_mean])
        print(mean)
        print(moving_mean)
        print(update_moving_mean)









