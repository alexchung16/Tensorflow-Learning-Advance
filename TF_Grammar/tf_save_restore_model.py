#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File save_restore_model.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 7/11/2019 PM 20:18

import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

model_path = os.path.join(os.getcwd(), 'model')

def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.mkdir(path)


if __name__ == "__main__":

    model_name = 'test_model'
    save_path = os.path.join(model_path, model_name)

    # save model
    w1 = tf.Variable(tf.random_normal(shape=[2, 3]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[3, 2]), name='w2')
    m = tf.matmul(w1, w2)
    saver = tf.train.Saver(var_list=[w1, w2])
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, save_path=save_path, global_step=None)

    #restore model parameter
    with tf.Session() as sess:
        # sess.run(init)
        # print(sess.run(w1))
        new_saver = tf.train.import_meta_graph(os.path.join(model_path, 'test_model.meta'))
        new_saver.restore(sess, tf.train.latest_checkpoint(model_path + '/'))
        graph = tf.get_default_graph()
        print(graph.get_tensor_by_name('w1:0'))
        # new_saver.restore(sess, save_path=os.path.join(model_path, 'test_model.data-00000-of-00001'))
        # print(sess.run('w1:0'))

    # restore pretrain model
    w1 = tf.placeholder(dtype=tf.float32, shape=None, name='w1')
    w2 = tf.placeholder(dtype=tf.float32, shape=None, name='w2')
    bias = tf.Variable(initial_value=2.0, name='bias')
    with tf.name_scope('operation'):
        multi_op = tf.multiply(tf.add(w1, bias), w2, name='multiply_op')
    print(w1.name) # w1:0
    print(w2.name) # w2:0
    print(multi_op.name) # operation/multiply:0
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    with tf.Session() as sess:
        # initial variable
        sess.run(init)
        saver = tf.train.Saver()
        # construct feed dict
        feed_dict = {w1: 3,
                     w2: 4}
        multi = sess.run(fetches=[multi_op], feed_dict=feed_dict)
        print(multi)
        saver.save(sess, save_path=save_path, global_step=None)

    # restore graph from .meta
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_path, 'multiply_model.meta'))

        saver.restore(sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=model_path))
        print(sess.run('bias:0')) # 2.0

        # get graph
        graph = tf.get_default_graph()

        # get tensor from graph
        # operation = graph.get_operations()
        w1 = graph.get_tensor_by_name('w1:0')
        w2 = graph.get_tensor_by_name('w2:0')
        # multi_op = graph.get_operation_by_name('operation/multiply_op')
        multi_op = graph.get_tensor_by_name('operation/multiply_op:0')

        feed_dict = {w1: 4, w2: 5}
        multi_op = tf.cast(multi_op, dtype=tf.int32, name='convert_int32')
        pow_op = tf.pow(x=multi_op, y=2, name='pow')
        # (4 + 2) * 5 = 30
        multi, pow = sess.run(fetches=[multi_op, pow_op], feed_dict=feed_dict)
        print('multi_op:',  multi)
        print('pow_op:', pow)
    #
    # construct compute graph
    w1 = tf.placeholder(dtype=tf.float32, shape=None, name='w1')
    w2 = tf.placeholder(dtype=tf.float32, shape=None, name='w2')
    bias = tf.Variable(initial_value=0.0, name='bias')
    with tf.name_scope('operation'):
        multi_op = tf.multiply(tf.add(w1, bias), w2, name='multiply_op')
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        saver = tf.train.Saver()
        # initial bias
        print(sess.run('bias:0'))
        saver.restore(sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=model_path))

        print(sess.run('bias:0'))
        print(tf.compat.v1.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=None)) # [<tf.Variable 'bias:0' shape=() dtype=float32_ref>]

        # # get tensor by get_collection
        # bias = tf.get_collection(key='w1')
        # print(bias)
        # get tensor by graph
        bias = graph.get_tensor_by_name(name='bias:0')
        print(bias)





























