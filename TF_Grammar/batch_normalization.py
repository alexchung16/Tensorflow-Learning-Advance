#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File batch_normalization.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 4/11/2019 PM 17:33

import os
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

def getVariable(name, shape, initializer, weight_decay=0.0, dtype=tf.float32, trainable=True):
    """
    add weight to tf.get_variable
    :param name:
    :param shape:
    :param initializer:
    :param weight_decay:
    :param dtype:
    :param trainable:
    :return:
    """
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    return tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, use_resource=regularizer,
                           trainable=trainable)


if __name__ == "__main__":
    # image test
    batch_size = 10
    height_size = 224
    width_size = 224
    features = 3
    bn_decay = 0.9
    variance_epsilon = 1e-3
    is_training = True
    m = tf.Variable(tf.random_uniform(shape=(2, 3), minval=0, maxval=2, dtype=tf.float32))
    img_batch = tf.random_uniform(shape=(10, 224, 224, 32))


    sess = tf.Session()
    with sess.as_default():
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init)

        # tf.nn.moments test
        # Calculate the mean and variance of x
        # mean, var = tf.nn.moments(x=m, axes=[0]) # (3,) (3,)
        # mean, var = tf.nn.moments(x=m, axes=[1]) # (2,) (2,)
        mean, var = tf.nn.moments(x=m, axes=[0, 1]) # () ()
        print(mean.get_shape(), var.get_shape()) #

        axis = list(range(len(img_batch.get_shape())-1)) # [0, 1, 2]
        # print(axis.eval())
        mean, variance = tf.nn.moments(x=img_batch, axes=axis, name='moment')
        print(mean.get_shape(), variance.get_shape()) # (32,) (32,)

        # tf.nn.batch_normalization test
        params_shape = tf.shape(img_batch)[-1:]
        beta = getVariable(name='beta', shape=params_shape, initializer=tf.zeros_initializer)
        gamma = getVariable(name='gamma', shape=params_shape, initializer=tf.ones_initializer)

        moving_mean = getVariable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = getVariable('moving_variance', params_shape, initializer=tf.zeros_initializer, trainable=False)
        # => variable * decay + value * (1 - decay) = 0
        # => variable -= (1 - decay) * (variable - value)
        moving_averages.assign_moving_average(variable=moving_mean, value=mean, decay=bn_decay)
        moving_averages.assign_moving_average(variable=moving_variance, value=variance, decay=bn_decay)

        mean, variance = control_flow_ops.cond(pred=is_training, true_fn=lambda:(mean, variance),
                                               false_fn=lambda:(moving_mean, moving_variance))
        x = tf.nn.batch_normalization(x=img_batch, mean=mean, variance=variance, offset=beta, scale=gamma,
                                      variance_epsilon=variance_epsilon, name='batch_normalization')


    sess.close()










