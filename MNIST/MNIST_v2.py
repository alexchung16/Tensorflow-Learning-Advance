#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File MNIST_v2.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 8/10/2019 PM 15:58


import os
import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

model_path = os.path.join(os.getcwd(), 'model')

def weightVariable(shape):
    """
    weight initialize variable filter/kernel

    :param shape:
    :return:
    """

    # tf.random_normal: return random values from a normal distribution
    # tf.truncated_normal: return random values from a normal distribution, except that values
    # whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked
    initial = tf.truncated_normal(shape=shape, mean=0., stddev=0.1, dtype=tf.float32, seed=0)
    return tf.Variable(initial)

def biasVariable(shape):
    """
    bias initialize constant
    :param shape:
    :return:
    """
    initial = tf.constant(shape=shape, value=0.1, dtype=tf.float32)
    return tf.Variable(initial)

def connv2d(input, kernel):
    """
    convolution 2D
    :param x: input tensor shape: [batch, in_height, in_width, in_channels]
    :param W: filter / kernel tensor of shape: [filter_height, filter_width, in_channels, out_channels]
    :return:
    """
    # tf.nn.conv2d: computes a 2-D convolution given 4-D input
    # stride: tride of the sliding window for each dimension of input, Must have strides[0] = strides[3] = 1
    # padding:  Either the string "SAME" or "VALID"
    return tf.nn.conv2d(input=input, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(input, pool_size):
    """
    max pooling
    :param pool_size:
    :return:
    """
    return tf.nn.max_pool2d(input=input, ksize=[1, pool_size[0], pool_size[1], 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # input data placeholder
    x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 28*28))
    # transform shape dims from 2 to 4
    # note: -1 equivalent to None,indicates an indefinite number.
    x_img = tf.reshape(tensor=x, shape=(-1, 28, 28, 1))

    # input target placeholder
    y_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 10))

    # create server to manage model
    
    # first convolution layer
    # get kernel weight and bias
    w_conv1 = weightVariable(shape=[5, 5, 1, 32])
    b_conv1 = biasVariable(shape=[32])
    # activation
    h_conv1 = tf.nn.relu(features=connv2d(x_img, w_conv1) + b_conv1)
    # pooling
    p_conv1 = maxPool(input=h_conv1, pool_size=[2, 2])

    # second convolution layer
    # get kernel weight and bias
    w_conv2 = weightVariable(shape=[5, 5, 32, 64])
    b_conv2 = biasVariable(shape=[64])
    # activation
    h_conv2 = tf.nn.relu(features=connv2d(p_conv1, w_conv2) + b_conv2)
    # pooling
    p_conv2 = maxPool(input=h_conv2, pool_size=(2, 2))


    # first full connect layers
    # 28/2/2=7
    # units = 1024
    w_fc1 = weightVariable(shape=[7*7*64, 1024])
    b_fc1 = biasVariable(shape=[1024])
    p_conv2_flat = tf.reshape(p_conv2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(features=tf.matmul(p_conv2_flat, w_fc1) + b_fc1)

    # dropout layer
    keep_prob = tf.placeholder(dtype=tf.float32)
    h_fc1_drop = tf.nn.dropout(x=h_fc1, keep_prob=keep_prob)

    # output layer
    w_fc2 = weightVariable(shape=[1024, 10])
    b_fc2 = biasVariable(shape=[10])

    y_conv = tf.nn.softmax(logits=tf.matmul(h_fc1_drop, w_fc2) + b_fc2)


    with tf.Session() as sess:
        # loss function
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        # optimize algorithm
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

        # evaluate model
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # accuracy mean
        # tf.reduce_mean: computes the mean
        # tf.cast: cast type
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # initialize variable
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob:1.0})
                print('step {0}: train accuracy {1}'.format(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})

        # evaluate model
        print("test accuracy {0}".format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})))





