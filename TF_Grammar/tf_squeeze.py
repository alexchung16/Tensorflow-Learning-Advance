#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File tf_squeeze.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 30/10/2019 AM 11:01

import numpy as np
import tensorflow as tf
import cv2 as cv

if __name__ == "__main__":
    tf.set_random_seed(seed=0)
    a = tf.random_uniform(shape=(2,1,1,2), minval=1, maxval=10 , dtype=tf.int32)
    with tf.Session() as sess:
        a = a.eval()
        print(a)
        s = tf.squeeze(input=a, axis=[1, 2])
        print(s.eval())

