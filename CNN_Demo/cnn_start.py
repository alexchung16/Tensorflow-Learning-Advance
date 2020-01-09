#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File cnn_start.py
# @ Description :
# @ Author alexchung
# @ Time 12/11/2019


import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from DataProcess.read_TFRecord import reader_tfrecord, get_num_samples

# origin dataset
original_dataset_dir = '/home/alex/Documents/datasets/dogs_vs_cat_separate'

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, 'test')

tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')
train_record = os.path.join(tfrecord_dir, 'image.tfrecords')
logs_dir = os.path.join(os.getcwd(), 'logs')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('height', 150, 'Number of height size.')
flags.DEFINE_integer('width', 150, 'Number of width size.')
flags.DEFINE_integer('depth', 3, 'Number of depth size.')
flags.DEFINE_integer('num_classes', 2, 'Number of image class.')
flags.DEFINE_integer('epoch', 30, 'Number of epoch.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('step_per_epoch', 100, 'Number of step size of per epoch.')
flags.DEFINE_float('keep_prop', 0.5, 'Number of probability that each element is kept.')

flags.DEFINE_string('train_dir', train_record, 'pretrain model dir.')

def cnn_net(inputs, keep_prob=0.5, is_training=True, scope='Cnn'):
    """

    :param inputs:
    :param scope:
    :return:
    """
    with tf.compat.v1.variable_scope(scope, default_name='cnn', values= [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='VALID'):
            # 150 x 150 x 3
            net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=(3, 3), stride=1,
                              activation_fn=tf.nn.relu, scope='Conv1_1a_3x3')
            # 148 x 148 x 32
            net = slim.max_pool2d(inputs=net, kernel_size=(2, 2), scope='MaxPool1_1a_2x2')

            # 74 x 74 x 32
            net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=(3, 3), stride=1,
                              activation_fn=tf.nn.relu, scope='Conv2_1a_3x3')
            # 32 x 32 x 64
            net = slim.max_pool2d(inputs=net, kernel_size=(2, 2), scope='MaxPool2_1a_2x2')

            # 36 x 36 x 64
            net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=(3, 3), stride=1,
                              activation_fn=tf.nn.relu, scope='Conv3_1a_3x3')
            # 34 x 34 x 128
            net = slim.max_pool2d(inputs=net, kernel_size=(2, 2), stride=2, scope='MaxPool3_1a_2x2')

            # 17 x 17 x 128
            net = slim.dropout(inputs=net, keep_prob=keep_prob, is_training=is_training, scope='dropout1')

            # 17 x 17 x 128
            net = slim.flatten(inputs=net, scope='flatten1')
            #
            net = slim.fully_connected(inputs=net, num_outputs=512, activation_fn=tf.nn.relu, scope='fcn1')
            #
            logits = slim.fully_connected(inputs=net, num_outputs=2, activation_fn=tf.nn.softmax, scope='softmax')

            return logits


if __name__ == "__main__":
    num_samples = get_num_samples(record_file=FLAGS.train_dir)
    batch_size = num_samples // FLAGS.step_per_epoch
    images, labels, filenames = reader_tfrecord(record_file=FLAGS.train_dir,
                                                batch_size=batch_size,
                                                input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                                                class_depth=FLAGS.num_classes,
                                                epoch=FLAGS.epoch,
                                                shuffle=False)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.height, FLAGS.width, FLAGS.depth], name='inputs')
    target = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.num_classes], name='labels')
    logits = cnn_net(inputs, keep_prob=FLAGS.keep_prop, is_training=True)
    # global_step = tf.train.create_global_step()
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    loss_op = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target))
    tf.summary.scalar(name='loss', tensor=loss_op)
    accuracy_op = tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=target, axis=1)), dtype=tf.int32)) / batch_size
    tf.summary.scalar(name='accuracy', tensor=accuracy_op)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    train_op = optimizer.minimize(loss=loss_op, global_step=global_step)


    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with tf.Session() as sess:

        # save operation graph
        write = tf.summary.FileWriter(logdir=logs_dir, graph=sess.graph)
        for var in tf.model_variables():
            print(var.name)
        # print(sess.run('Cnn/Conv1_1a_3x3/weights:0'))
        # print(sess.run('Cnn/Conv1_1a_3x3/biases:0'))
        # add weight and bias to logs
        graph = tf.get_default_graph()
        fcn1_weight = graph.get_tensor_by_name('Cnn/fcn1/weights:0')
        fcn1_bias = graph.get_tensor_by_name('Cnn/fcn1/biases:0')
        tf.summary.histogram('Cnn/fcn1/weights', fcn1_weight)
        tf.summary.histogram('Cnn/fcn1/biases', fcn1_bias)
        # merge summary operation
        summary_op = tf.summary.merge_all()

        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if not coord.should_stop():
                for epoch in range(FLAGS.epoch):
                    print('Epoch: {0}/{1}'.format(epoch, FLAGS.epoch))
                    for step in range(FLAGS.step_per_epoch):

                        image, label, filename = sess.run([images, labels, filenames])

                        feed_dict = {inputs: image, target: label}

                        _, loss_value, train_accuracy, summary = sess.run(
                            fetches=[train_op, loss_op, accuracy_op, summary_op],
                            feed_dict=feed_dict)

                        print('  Step {0}/{1}: loss value {2}  train accuracy {3}'
                              .format(step, FLAGS.step_per_epoch, loss_value, train_accuracy))

                        # add summary protocol buffer to the event file
                        write.add_summary(summary, global_step=epoch * FLAGS.epoch + step)
                write.close()

        except Exception as e:
            print(e)

        coord.request_stop()
    coord.join(threads)
    sess.close()
    print('model training has complete')










