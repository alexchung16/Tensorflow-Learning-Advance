#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File tf_squeeze.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 30/10/2019 AM 11:58

import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    tf.set_random_seed(seed=0)
    a = tf.random_uniform(shape=(3,3), minval=1, maxval=10 , dtype=tf.float32)
    with tf.Session() as sess:
        a = a.eval()
        print(a)
        s = tf.nn.softmax(logits=a, axis=-1)
        print(s.eval())