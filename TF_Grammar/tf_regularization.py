#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_regularization.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/13 下午3:03
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf


def fully_connected(input_op, scope, num_outputs, weight_decay=0.00004, is_activation=True, fineturn=True):
    """
     full connect operation
    :param input_op:
    :param scope:
    :param num_outputs:
    :param parameter:
    :return:
    """
    # get feature num
    shape = input_op.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    with tf.compat.v1.variable_scope(scope):
        flat_data = tf.reshape(tensor=input_op, shape=[-1, size], name='Flatten')

        weights =get_weights_1(shape=[size, num_outputs], weight_decay=weight_decay, trainable=fineturn)
        biases =get_bias(shape=[num_outputs], trainable=fineturn)

        if is_activation:
             return tf.nn.relu_layer(x=flat_data, weights=weights, biases=biases)
        else:
            return tf.nn.bias_add(value=tf.matmul(flat_data, weights), bias=biases)


def get_bias(shape, trainable=True):
    """
    get bias
    :param bias_shape:

    :return:
    """
    bias = tf.get_variable(shape=shape, name='Bias', dtype=tf.float32, trainable=trainable)

    return bias


def what_regularization():
    """
    what regularization is doing
    :return:
    """
    # L2 amounts to adding a penalty on the norm of the weights to the loss.
    # reference https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
    weight_decay = 0.00004
    weight = tf.Variable(initial_value=tf.constant(value=[1.0, 2.0, 3.0]))

    # use tensorflow interface
    weight_loss_1 = tf.nn.l2_loss(weight) * weight_decay
    weight_loss_2 = tf.contrib.layers.l2_regularizer(scale=weight_decay)(weight)

    # cunstom
    custom_weight_loss = tf.reduce_sum(tf.multiply(weight, weight))
    custom_weight_loss = 1 / 2 * weight_decay * custom_weight_loss

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    print(sess.run(weight_loss_1))
    print(sess.run(weight_loss_2))
    print(sess.run(custom_weight_loss))


def get_weights_1(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
    """
    add weight to tf.get_variable
    :param name:
    :param shape:
    :param initializer:
    :param weight_decay:
    :param dtype:
    :param trainable:
    :return:
    """
    # create regularizer
    if weight_decay > 0:
        regularizer= tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    weight = tf.get_variable(name='Weights', shape=shape, dtype=dtype, regularizer=regularizer,
                              trainable=trainable)
    return weight


def get_weights_2(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
    """
    add weight regularization to loss collection
    :param name:
    :param shape:
    :param initializer:
    :param weight_decay:
    :param dtype:
    :return:
    """
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.01), name='Weights', dtype=dtype,
                         trainable=trainable)
    if weight_decay > 0:
        weight_loss = tf.contrib.layers.l2_regularizer(weight_decay)(weight)
        # weight_loss = tf.nn.l2_loss(weight, name="weight_loss")
        # tf.add_to_collection(tf.GraphKeys.LOSSES, value=weight_loss)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
    else:
        pass

    return weight

def get_weights_3(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
    """

    :param name:
    :param shape:
    :param weight_decay:
    :param dtype:
    :return:
    """
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.01), name='Weights', dtype=dtype,
                         trainable=trainable)
    if weight_decay > 0:
        weight_loss = tf.contrib.layers.l2_regularizer(weight_decay)(weight)
        # weight_loss = tf.nn.l2_loss(weight, name="weight_loss")
        # tf.add_to_collection("weight_loss", value=weight_loss)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
    else:
        pass

    return weight


def model_nets(input_batch, num_classes=None, weight_decay=0.00004, scope="test_nets"):
    """
    demo model
    :param image_shape:
    :param num_classes:
    :param weight_deca:
    :return:
    """
    with tf.variable_scope(scope):
        net = fully_connected(input_batch, num_outputs=128, weight_decay=weight_decay, scope='fc1')
        net = fully_connected(net, num_outputs=32, weight_decay=weight_decay, scope='fc2')
        net = fully_connected(net, num_outputs=num_classes, is_activation=False, weight_decay=weight_decay, scope='logits')
        prob = tf.nn.softmax(net, name='prob')
    return prob


if __name__ == "__main__":

    # 函数说明
    # tf.add_to_collection： 将一个变量加入到集合(collection)中
    # tf.get_collection： 从一个集合(collection)取出所有变量
    # tf.add_n：把一个列表的所有变量相加

    # step 1 what regularization is doing
    # what_regularization()

    # step how to use regularization in network

    BATCH_SIZE = 10
    DATA_LENGTH = 1024
    NUM_CLASSES = 5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.00004

    # construct net

    global_step = tf.train.get_or_create_global_step()
    input_data_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, DATA_LENGTH], name="input_data")
    input_label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name="input_label")
    # inference part
    logits = model_nets(input_batch=input_data_placeholder, num_classes=NUM_CLASSES, weight_decay=WEIGHT_DECAY)

    # calculate loss part
    with tf.variable_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_label_placeholder,
                                                                name='entropy')
        loss_op = tf.reduce_mean(input_tensor=cross_entropy, name='loss')
        weight_loss_op = tf.losses.get_regularization_losses()
        weight_loss_op = tf.add_n(weight_loss_op)
        total_loss_op = loss_op + weight_loss_op

    # generate data and label
    tf.set_random_seed(0)
    data_batch = tf.Variable(tf.random_uniform(shape=(BATCH_SIZE, DATA_LENGTH), minval=0, maxval=1, dtype=tf.float32))
    label_batch = tf.Variable(tf.random_uniform(shape=(BATCH_SIZE,), minval=1, maxval=NUM_CLASSES, dtype=tf.int32))
    label_batch = tf.one_hot(label_batch, depth=NUM_CLASSES) # convert label to onehot


    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        input_data, input_label = sess.run([data_batch, label_batch])
        # print(input_data)
        # print(input_label)
        for var in tf.global_variables():
            print(var.op.name, var.shape)
        print(tf.trainable_variables())

        # training part
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss=total_loss_op,
                                                                                           global_step=global_step)

        feed_dict = {input_data_placeholder:input_data,
                     input_label_placeholder:input_label}

        _, total_loss, loss, weight_loss = sess.run([train_op, total_loss_op, loss_op, weight_loss_op],
                                                             feed_dict=feed_dict)
        print('loss:{0} weight loss:{1} total loss:{1}'.format(loss, weight_loss, total_loss))











