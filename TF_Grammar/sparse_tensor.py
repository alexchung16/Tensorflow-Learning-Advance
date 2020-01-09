#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File sparse_tensor.py
# @ Description :
# @ Author alexchung
# @ Time 10/10/2019 PM 19:13

import os
import tensorflow as tf

if __name__ == "__main__":
    sp = tf.SparseTensor(indices=[[0, 0], [0, 2], [1, 1]], values=[1 , 1, 1], dense_shape=[2, 3])
    with tf.Session() as sess:
        print(sp.eval())
        print(tf.sparse_reduce_sum(sp_input=sp).eval())
        print(tf.sparse_reduce_sum(sp_input=sp, axis=0, keep_dims=True).eval())
        print(tf.sparse_reduce_sum(sp_input=sp, axis=1, keep_dims=True).eval())
        print(tf.sparse_reduce_sum(sp_input=sp, axis=[0, 1]).eval())