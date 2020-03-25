#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File batch_normalization.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 4/11/2019 PM 17:33

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

def get_variable(name, shape, initializer, weight_decay=0.0, dtype=tf.float32, trainable=True):
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
        regularize= tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularize = None
    return tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, use_resource=regularize,
                           trainable=trainable)


if __name__ == "__main__":
    # image test
    batch_size = 10
    height_size = 224
    width_size = 224
    features = 3
    bn_decay = 0.9
    variance_epsilon = 1e-3
    is_training = tf.convert_to_tensor(value=True, dtype=bool)
    m = tf.Variable(tf.random_uniform(shape=(2, 3), minval=0, maxval=2, dtype=tf.float32))
    img_batch = np.random.uniform(size=(10, 224, 224, 32)).astype(np.float32)

    # ---------------------------------BN(Batch Normalization)----------------------------------------
    # tf.nn.moments test
    # Calculate the mean and variance of x
    # mean, var = tf.nn.moments(x=m, axes=[0]) # (3,) (3,)
    # mean, var = tf.nn.moments(x=m, axes=[1]) # (2,) (2,)
    # mean, var = tf.nn.moments(x=m, axes=[0, 1])  # () ()
    # print('mean: {0}  var: {1}a'.format(mean.eval(), var.eval()))  #

    # tf.nn.batch_normalization test
    params_shape = img_batch.shape[-1:]
    beta = get_variable(name='beta', shape=params_shape, initializer=tf.zeros_initializer)
    gamma = get_variable(name='gamma', shape=params_shape, initializer=tf.ones_initializer)

    moving_mean = get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = get_variable('moving_variance', params_shape, initializer=tf.zeros_initializer,
                                   trainable=False)

    axis = list(range(len(img_batch.shape) - 1))  # [0, 1, 2]
    # print(axis.eval())
    mean, variance = tf.nn.moments(x=tf.cast(img_batch, dtype=tf.float32), axes=axis, name='moment')
    print('mean: {0}  variance: {1}'.format(mean, variance))

    # => variable * decay + value * (1 - decay) = 0
    # => variable -= (1 - decay) * (variable - value)
    moving_averages.assign_moving_average(variable=moving_mean, value=mean, decay=bn_decay)
    moving_averages.assign_moving_average(variable=moving_variance, value=variance, decay=bn_decay)

    mean, variance = control_flow_ops.cond(pred=is_training, true_fn=lambda: (mean, variance),
                                           false_fn=lambda: (moving_mean, moving_variance))

    bn_image = tf.nn.batch_normalization(x=img_batch, mean=mean, variance=variance, offset=beta, scale=gamma,
                                         variance_epsilon=variance_epsilon, name='batch_normalization')
    sess = tf.Session()
    with sess.as_default():
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init)

        bn_image = sess.run(bn_image)
        print(img_batch, bn_image)

    sess.close()










