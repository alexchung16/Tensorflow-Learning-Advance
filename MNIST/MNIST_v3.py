#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File MNIST_v3.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 9/10/2019 AM 09:34


from __future__ import print_function
import os
import math
import numpy as py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 40000, 'Number of steps to run trainer.')
flags.DEFINE_integer('input_size', 28*28, 'Number of input size.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('class_size', 10, 'Number of image class.')
flags.DEFINE_integer('batch_size', 100, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'MNIST_data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')


def placeholder_inputs(batch_size):
    """
    Generate placeholder variables to represent the input tensors
    :param batch_size:
    :return:
    """
    images_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, FLAGS.input_size))
    # label_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.class_size))
    label_placeholder = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size))
    return images_placeholder, label_placeholder



def fill_feed_dict(data_set, images_pl, labels_pl):
    """
    Fills the feed_dict for training the given step
    :param data_set:
    :param images_pl:
    :param labels_pl:
    :return:
    """
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def weightVariable(shape):
    """
    weight initialize variable filter/kernel

    :param shape:
    :return:
    """

    initial = tf.truncated_normal(shape=shape, mean=0., stddev=1.0/math.sqrt(float(shape[0])), dtype=tf.float32, seed=0)
    return tf.Variable(initial, name='weight')

def biasVariable(shape):
    """
    bias initialize constant
    :param shape:
    :return:
    """
    initial = tf.zeros(shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name='bias')


def full_connect_layer(name_scope, input_tensor, shape, activation = 'relu'):
    """
    full connect layer
    :param name_scope:
    :param input_tensor: input tensor
    :param shape: (input_units, layer_units)
    :param activation:
    :return:
    """
    with tf.name_scope(name_scope):
        weight = weightVariable(shape)

        bias = biasVariable([shape[1]])
        if activation == 'relu':
            return tf.nn.relu(tf.matmul(input_tensor, weight) + bias)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(tf.matmul(input_tensor, weight) + bias)
        elif activation == 'softmax':
            return tf.nn.softmax(tf.matmul(input_tensor, weight) + bias)


def mnist_fcn_model(image_input):
    """
    mnist full connect network
    :param image_input: input placeholder
    :return: 
    """
    # hidden1 layer
    h_fc1 = full_connect_layer(name_scope='hidden1', input_tensor=image_input,
                               shape=[FLAGS.input_size, FLAGS.hidden1], activation='relu')
    # hidden2 layer
    h_fc2 = full_connect_layer(name_scope='hidden2', input_tensor=h_fc1,
                               shape=[FLAGS.hidden1, FLAGS.hidden2], activation='relu')
    # softmax layer
    logits = full_connect_layer(name_scope='softmax_linear', input_tensor=h_fc2,
                                  shape=[FLAGS.hidden2, FLAGS.class_size], activation='softmax')

    return logits


def loss(logits, labels):
    """
    loss function
    :param logits:
    :param labels:
    :return:
    """
    # cross_entropy = -tf.reduce_sum(labels * tf.log(logits))
    # return cross_entropy
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    """
    train function
    :param loss:
    :param learning_rate:
    :return:
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
  """
  Evaluate the quality of the logits at predicting the label.
  :param logits:
  :param labels:
  :return:
  """
  # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    """
    Runs one evaluation against the full epoch of data.
    :param sess:
    :param eval_correct:
    :param images_placeholder:
    :param labels_placeholder:
    :param data_set:
    :return:
    """

    true_count = 0
    # epoch num
    step_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = step_per_epoch * FLAGS.batch_size
    for n in range(step_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(fetches=eval_correct, feed_dict=feed_dict)

    precision = true_count / num_examples
    print('Num examples: {0} Num correct: {1} Precision @ 1: {2}'.format(num_examples, true_count, precision))

def running_train():

    mnist = input_data.read_data_sets(train_dir='MNIST_data', one_hot=FLAGS.fake_data)

    # with tf.Session() as sess:
    #
    #     print(mnist.train.labels)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # input placeholder
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        # model
        logits = mnist_fcn_model(images_placeholder)
        # loss
        losses = loss(logits, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(losses, FLAGS.learning_rate)
        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # summary add loss
        tf.summary.scalar("loss", losses)
        # 状态可视化
        # 为了释放TensorBoard所使用的event file, 所有的即时数据都要到图表构建阶段合并至一个op中
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        # Instantiate a SummaryWriter to output summaries and the Graph.
        saver = tf.train.Saver()

        # optimize algorithm
        # train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

        with tf.Session() as sess:
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(logdir=FLAGS.train_dir, graph=sess.graph_def)
            # initial global variable
            sess.run(init)
            for step in range(FLAGS.max_steps):
                feed_dict = fill_feed_dict(data_set=mnist.train, images_pl=images_placeholder, labels_pl=labels_placeholder)
                _, loss_value,  train_accuracy = sess.run([train_op, losses, eval_correct], feed_dict=feed_dict)
                # train_accuracy = eval_correct.eval(feed_dict=feed_dict)
                if step % 100 == 0:
                    print('step {0}: loss value={1} train accuracy {2}%'.format(step, loss_value, train_accuracy))
                    # add train data to log
                    summary_str = sess.run(fetches=summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary=summary_str, global_step=step)
                    summary_writer.flush()
                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    # get save path
                    checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                    # 保存当前检查点（check point）/步骤的所有可训练变量值到 check point 文件
                    saver.save(sess=sess, save_path=checkpoint_file, global_step=step)

            # Evaluate against the training set
            print('train data accuracy')
            do_eval(sess, eval_correct,images_placeholder,labels_placeholder, mnist.train)
            print('validation data accuracy')
            do_eval(sess, eval_correct, images_placeholder, labels_placeholder, mnist.validation)
            print('test data accuracy')
            do_eval(sess, eval_correct, images_placeholder, labels_placeholder, mnist.test)


if __name__ == "__main__":

    running_train()