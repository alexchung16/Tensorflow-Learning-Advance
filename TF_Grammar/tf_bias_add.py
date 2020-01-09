#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File bias_add.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 6/11/2019 16:16


import os
import tensorflow as tf


if __name__ == "__main__":


    num_out = 2
    img_data = tf.Variable(initial_value=tf.random_uniform(shape=(4, 3), minval=0, maxval=3, dtype=tf.float32))
    features = img_data.get_shape()[-1].value
    weight = tf.Variable(initial_value=tf.constant(value=1.0, shape=[features, num_out], dtype=tf.float32))
    bias = tf.Variable(initial_value=tf.constant(value=1.0, shape=[num_out], dtype=tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        img_data, weight,  bias = sess.run([img_data, weight, bias])
        print(img_data, bias)
        # test tf.nn.bias_add
        # out_data_0 = tf.nn.bias_add(value=img_data, bias=bias)
        # print(out_data_0.eval())

        # test tf.nn.relu_layer
        out_data_1 = tf.matmul(img_data, weight)
        out_data_2 = tf.nn.bias_add(out_data_1, bias)
        out_data_3 = tf.nn.relu(out_data_2)

        out_data_4 = tf.nn.relu_layer(x=img_data, weights=weight, biases=bias)
        assert out_data_3 == out_data_3
        print(out_data_3.eval())
        print(out_data_3.eval())










