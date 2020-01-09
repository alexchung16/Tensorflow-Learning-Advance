#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File batch_normalization.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 4/11/2019 PM 14:31

import os
import tensorflow as tf

if __name__ == "__main__":
    shape = (32, )
    a = tf.get_variable(name='a', shape=shape, initializer=tf.ones_initializer)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(a.eval())