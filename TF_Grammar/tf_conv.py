#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_conv.py
# @ Description:
# @ Reference  : https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
#                https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#filter
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/25 下午2:07
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d

# image_dir = '../Dataset/Image'
# video_dir = '../Dataset/Video'
#
# flower_img = os.path.join(image_dir, 'sunflower.jpg')
# traffic_video = os.path.join(video_dir, 'traffic.mp4')

def conv3d_layer(inputs, output_channels, kernel_size=None, strides=None, use_bias=False, padding='SAME', name=None,
                 xavier=False):
    """
     conv2d layer
    :param inputs:
    :param filters: num output channels of per frame
    :param kernel_size: [filter_depth, filter_height,filter_width]
    :param strides:
    :param use_bias:
    :param padding:
    :param scope:
    :param xavier:
    :return:
    """
    if kernel_size is None:
         kernel_size = [3, 3, 3]
    if strides is None:
        strides = [1, 1, 1]
        # get feature num
    input_channels = inputs.get_shape()[-1].value
    with tf.compat.v1.variable_scope(name):

        # filters [depth, height, width, in_channels, output_channels]
        filters = get_filter(filter_shape=[kernel_size[0], kernel_size[1], kernel_size[2], input_channels, output_channels],
                             name=name, xavier=xavier)

        outputs = tf.nn.conv3d(input=inputs, filter=filters, strides=[1, strides[0], strides[1], strides[2], 1],
                               name=name, padding=padding)

        if use_bias:
            biases = get_bias(bias_shape=[output_channels], name=name, xavier=xavier)
            outputs = tf.nn.bias_add(value=outputs, bias=biases)

        return tf.nn.relu(outputs)



def conv2d_layer(inputs, output_channels, kernel_size=None, strides=None, use_bias=False, padding='SAME', name=None,
                 xavier=False):
    """
     conv2d layer
    :param inputs:
    :param filters: num output channels of per frame
    :param kernel_size: [filter_height,filter_width]
    :param strides:
    :param use_bias:
    :param padding:
    :param scope:
    :param xavier:
    :return:
    """
    if kernel_size is None:
         kernel_size = [3, 3]
    if strides is None:
        strides = 1
        # get feature num
    input_channels = inputs.get_shape()[-1].value
    with tf.compat.v1.variable_scope(name):
        # filters [filter_height, filter_width, in_channels, out_channels]
        filters = get_filter(filter_shape=[kernel_size[0], kernel_size[1], input_channels, output_channels], name=name,
                            xavier=xavier)

        outputs = tf.nn.conv2d(input=inputs, filter=filters, strides=[1, strides, strides, 1], name=name,
                               padding=padding)

        if use_bias:
            biases = get_bias(bias_shape=[output_channels], name=name, xavier=xavier)
            outputs = tf.nn.bias_add(value=outputs, bias=biases)

        return tf.nn.relu(outputs)



def get_filter(filter_shape, xavier=False, name=None):
    """
    convolution layer filter
    :param filter_shape:
    :return:
    """
    with tf.variable_scope(name) as scope:
        if xavier:
            filter = tf.get_variable(name='weights', shape=filter_shape, initializer=xavier_initializer_conv2d(),
                                     trainable=True, dtype=tf.float32)
        else:
            filter = tf.Variable(initial_value=tf.random.truncated_normal(shape=filter_shape, mean=0.0, stddev=1e-1, dtype=tf.float32),
                                 trainable=True, name='weights')
        return filter



def get_bias(bias_shape, xavier=False, name=None):
    """
    get bias
    :param bias_shape:
    :return:
    """
    with tf.variable_scope(name) as scope:
        if xavier:
            bias = tf.get_variable(name='biases', shape=bias_shape, initializer=xavier_initializer(), trainable=True,
                                   dtype=tf.float32)
        else:
            bias = tf.Variable(initial_value=tf.constant(value=0.1, shape=bias_shape, dtype=tf.float32),
                               trainable=True, name='biases')
        return bias

if __name__ == "__main__":

    BATCH_SIZE = 1
    IMAGE_SIZE = 224
    IMAGE_CHANNEL = 3
    FLOW_IMAGE_CHNNEL = 2
    FRAME_SIZE = 64

    np.random.seed(0)
    image = np.random.randint(low=1, high=255, size=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))
    rgb_video = np.random.randint(low=1, high=255, size=(BATCH_SIZE, FRAME_SIZE, IMAGE_SIZE, IMAGE_SIZE,
                                                         IMAGE_CHANNEL))
    flow_video = np.random.randint(low=1, high=255, size=(BATCH_SIZE, FRAME_SIZE, IMAGE_SIZE, IMAGE_SIZE,
                                                          FLOW_IMAGE_CHNNEL))

    image_input = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL],
                               name='rgb_input')
    video_input = tf.placeholder(dtype=tf.float32, shape=[None, FRAME_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL],
                               name='flow_input')

    # padding VALID or SAME

    # "VALID" only ever drops the right-most columns (or bottom-most rows).
    # "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd,
    # it will add the extra column to the right

    # For the SAME padding, the output height and width are computed as:
    # out_height = ceil(float(in_height) / float(strides[1]))
    # out_width  = ceil(float(in_width) / float(strides[2]))

    # For the VALID padding, the output height and width are computed as:
    # out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    # out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

    with tf.variable_scope('conv2d_net'):
        # 224x224x3
        net_0 = conv2d_layer(inputs=image_input, output_channels=32, kernel_size=[3, 3], strides=2, padding='SAME',
                           name='conv2d_0a_3x3')
        net_1 = tf.nn.max_pool2d(input=net_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               name='maxpool2d_2x2')
        net_2 = conv2d_layer(inputs=net_1, output_channels=64, kernel_size=[3, 3], strides=2, padding='VALID',
                           name='conv2d_0b_3x3')

    with tf.variable_scope('conv3d_net'):
        # 64x224x224x3
        net_3 = conv3d_layer(inputs=video_input, output_channels=32, kernel_size=[3, 3, 3], strides=[2, 2, 2],
                             padding='SAME', name='conv3d_0a_3x3')
        net_4 = tf.nn.max_pool3d(input=net_3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID',
                               name='maxpool3d_2x2')
        net_5 = conv3d_layer(inputs=net_4, output_channels=64, kernel_size=[3, 3, 3], strides=[2, 2, 2], padding='VALID',
                           name='conv3d_0b_3x3')

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        feed_dict_image = {image_input: image}
        net_0, net_1, net_2 = sess.run([net_0, net_1, net_2], feed_dict=feed_dict_image)

        # 224 x 224 x 3
        print(net_0.shape) # (1, 112, 112, 32)
        # ceil(224 / 2) = 112
        # 112 x 112 x 32
        print(net_1.shape) # (1, 56, 56, 32)
        # ceil((112- 2 + 1) / 2) = 56
        # 56 x 56 x 32
        print(net_2.shape) # (1, 27, 27, 64)
        # ceil((56 -3 +1) / 2) = 27
        # 27 x 27 x 64

        feed_dict_video = {video_input: rgb_video}
        net_3, net_4, net_5 = sess.run([net_3, net_4, net_5], feed_dict=feed_dict_video)
        # 64 X 224 X 224 X 3
        print(net_3.shape)  # (1, 32, 112, 112, 32)
        # ceil(64 / 2) = 32
        # ceil(224 / 2) = 112
        # 32 X 112 X 1X2 X 3
        print(net_4.shape)  # (1, 16, 56, 56, 32)
        # ceil((32- 2 + 1) / 2) = 16
        # ceil((112- 2 + 1) / 2) = 56
        # 16 X 56 x 56 x 32
        print(net_5.shape)  # (1, 7, 27, 27, 64)
        # ceil((16 -3 +1) / 2) = 7
        # ceil((56 -3 +1) / 2) = 27
        # 7 X 27 X 27





