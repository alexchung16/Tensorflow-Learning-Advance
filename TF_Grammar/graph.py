#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File graph.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 19/10/2019 AM 11:09

import os
import numpy
import tensorflow as tf


if __name__ == "__main__":
    # graph
    n = tf.constant(value=[0, 1, 2], dtype=tf.float32)
    assert n.graph is tf.get_default_graph()
    print('n is using default graph')
    # create new graph
    m = tf.Graph()
    # change m as default graph
    with m.as_default():
        assert m is tf.get_default_graph()
        print('m change to default graph')
        assert n.graph is not tf.get_default_graph()
        print('n graph is not m')
