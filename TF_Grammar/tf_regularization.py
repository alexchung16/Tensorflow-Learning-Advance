#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_regularization.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/13 下午3:03
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf


def what_regularization():
    """
    what regularization is doing
    :return:
    """
    # L2 amounts to adding a penalty on the norm of the weights to the loss.
    # reference https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
    weight_decay = 0.00004
    weight = tf.Variable(initial_value=tf.constant(value=[1.0, 2.0, 3.0]))

    # use tensorflow interface
    weight_loss_1 = tf.nn.l2_loss(weight) * weight_decay
    weight_loss_2 = tf.contrib.layers.l2_regularizer(scale=weight_decay)(weight)

    # cunstom
    custom_weight_loss = tf.reduce_sum(tf.multiply(weight, weight))
    custom_weight_loss = 1 / 2 * weight_decay * custom_weight_loss

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    print(sess.run(weight_loss_1))
    print(sess.run(weight_loss_2))
    print(sess.run(custom_weight_loss))


def get_weights_1(name, shape, initializer, weight_decay=0.0, dtype=tf.float32, trainable=True):
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
    # create regularizer
    if weight_decay > 0:
        regularizer= tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    weight = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer,
                              trainable=trainable)
    return weight


def get_weights_2(name, shape, weight_decay=0.0, dtype=tf.float32):
    """
    add weight regularization to loss collection
    :param name:
    :param shape:
    :param initializer:
    :param weight_decay:
    :param dtype:
    :return:
    """
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.01), name=name, dtype=dtype)
    if weight_decay > 0:
        weight_loss = tf.contrib.layers.l2_regularizer(weight_decay)(weight)
        # weight_loss = tf.nn.l2_loss(weight, name="weight_loss")
        # tf.add_to_collection(tf.GraphKeys.LOSSES, value=weight_loss)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
    else:
        pass

    return weight

def get_weights_3(name, shape, weight_decay=0.0, dtype=tf.float32):
    """

    :param name:
    :param shape:
    :param weight_decay:
    :param dtype:
    :return:
    """
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.01), name=name, dtype=dtype)
    if weight_decay > 0:
        weight_loss = tf.contrib.layers.l2_regularizer(weight_decay)(weight)
        # weight_loss = tf.nn.l2_loss(weight, name="weight_loss")
        # tf.add_to_collection("weight_loss", value=weight_loss)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
    else:
        pass

    return weight

if __name__ == "__main__":

    # 函数说明
    # tf.add_to_collection： 将一个变量加入到集合(collection)中
    # tf.get_collection： 从一个集合(collection)取出所有变量
    # tf.add_n：把一个列表的所有变量相加

    # step 1 what regularization is doing
    what_regularization()

    # step how to use regularization in network



