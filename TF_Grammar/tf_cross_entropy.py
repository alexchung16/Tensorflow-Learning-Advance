#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_cross_entropy.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/21 下午5:48
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow as tf

logits = tf.Variable(initial_value=[[0.2, 0.5, 0.3],
                                     [0.4, 0.2, 0.4],
                                     [0.5, 0.1, 0.4]])
labels = tf.Variable([1, 2, 0])
labels = tf.one_hot(labels, depth=3)


softmax_predict = tf.nn.softmax(logits, axis=-1)


def softmax_cross_entropy(labels, logits):
    """
    :param labels: (batch_size, label_depth)
    :param logits: (batch_size, label_depth)
    :return:
    """
    softmax_logits = tf.nn.softmax(logits, axis=-1)
    log_logits = tf.log(softmax_logits)
    loss = - tf.reduce_sum(tf.multiply(labels, log_logits), axis=-1)

    return loss

def sigmoid_cross_entropy(logits=None, labels=None):
    """
    x = logits, z = labels
    loss = z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
         = x - x * z + log(1 + exp(-x))
    :param label:
    :param logits:
    :return:
    """

    return logits - tf.multiply(logits, labels) + tf.log(sigmoid_reciprocal(logits))


def sigmoid_reciprocal(inputs):
    """
    (1 + exp(-x))
    :param inputs:
    :return:
    """
    return tf.add(1.0,  tf.exp(-inputs))


def label_smoothing(inputs, epsilon=0.1):
    """
    label smoothing
    see. https://arxiv.org/abs/1906.02629
    :param inputs: input_label
    :param epsilon: smoothing rate
    :return:
    """
    v = inputs.get_shape().as_list()[-1]
    return tf.multiply(1.0-epsilon, inputs) + tf.div(epsilon, v)

smooth_labels = label_smoothing(labels)

if __name__ == "__main__":


    loss_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss_softmax = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    custom_sigmoid_loss = sigmoid_cross_entropy(labels=labels, logits=logits)
    custom_softmax_loss = softmax_cross_entropy(labels=labels, logits=logits)


    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        print(labels.eval())
        print(smooth_labels.eval())

        assert loss_sigmoid.eval().all() == custom_sigmoid_loss.eval().all()
        print(loss_sigmoid.eval())
        print(custom_sigmoid_loss.eval())

        assert softmax_predict.eval().all() == custom_softmax_loss.eval().all()
        print(softmax_predict.eval())
        print(loss_softmax.eval())
        print(custom_softmax_loss.eval())

        print('Done!')







