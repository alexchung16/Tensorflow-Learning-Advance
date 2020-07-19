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


    # part 1 test assgin operation
    a = tf.Variable(initial_value=tf.random_uniform(shape=(3, 2), minval=0, maxval=3))
    b = tf.Variable(initial_value=tf.random_uniform(shape=(2, 3), minval=0, maxval=3))
    # tf.assign
    # tf.assign value to ref
    a_ = tf.assign(ref=a, value=b, validate_shape=False)

    # part 2 test moving average operation
    tf.random.set_random_seed(0)
    input = tf.Variable(initial_value=tf.random_uniform(shape=(2, 3), minval=0, maxval=3, dtype=tf.int32))
    axis = list(range(len(input.get_shape()) - 1))
    mean, variance = tf.nn.moments(x=tf.cast(input, tf.float32), axes=axis, name='moment')
    params_shape = input.get_shape()[-1:]
    moving_mean = tf.get_variable(name='moving_mean', shape=params_shape, initializer=tf.ones_initializer, trainable=False)

    update_moving_mean = moving_averages.assign_moving_average(variable=moving_mean, value=mean, decay=0.9)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)
        print("a before assign:\n", a.eval())
        print("b before assign:\n", b.eval())
        print("a after assign operation: \n", a_.eval())
        print('*' * 40)


        # assign_moving_average
        print('input value:\n', input.eval())
        print('mean :', mean.eval())
        print('variance:', variance.eval())

        print('moving mean:', moving_mean.eval())
        print("update moving mean:", update_moving_mean.eval())









