#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File tf_assign.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 7/11/2019 PM 20:49


import os
import tensorflow as tf

if __name__ == "__main__":
    w = tf.Variable(initial_value=0.0, name='w')
    t = tf.multiply(w, 2.0)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(t.eval())
        sess.run(tf.assign_add(w, 2.0))
        print(t.eval())
    print(os.listdir(' /home/alex/Documents/datasets/dogs_and_cat_separate/train'))

