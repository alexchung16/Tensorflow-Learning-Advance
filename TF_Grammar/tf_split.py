#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File tf_split.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 1/12/2019 AM 10:15

import tensorflow as tf


means = [123.68, 116.78, 103.94]

if __name__ == "__main__":


    init = tf.global_variables_initializer()
    batch_image = tf.constant(shape=[5, 224, 224, 3], value=1., dtype=tf.float32)
    with tf.Session() as sess:
        sess.run(init)

        num_channels = batch_image.get_shape()[-1]
        print(batch_image.eval())
        channels = tf.split(value=batch_image, num_or_size_splits=num_channels, axis=3)
        for n in range(num_channels):
            print("------------------------------------------------------------------")
            channels[n] -= means[n]
        batch_mean_image = tf.concat(values=channels, axis=3)
        print("------------------------------------------------------------------")
        print(batch_mean_image.eval())
        # print(batch_mean_image.eval())

