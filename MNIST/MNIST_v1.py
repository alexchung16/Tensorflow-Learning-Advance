#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File MNIST_v1.py
# @ Description : net only contain one full connect layer
# @ Author alexchung
# @ Time 8/10/2019 AM 09:24


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


dataset_path = '/home/alex/Documents/datasets/'
mnist_path = os.path.join(dataset_path, 'mnist.npz')

def load_mnist_data(data_path):
    """
    load mnist data
    :param data_path:
    :return:
    """
    f = np.load(data_path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def dense_to_onehot(label_dense, num_classes=None, dtype='float32'):
    """
    convert class label from scalar to one-hot vector
    :param label_dense:

    :param num_class:
    :param type:
    :return:
    """
    label_dense = np.array(label_dense, dtype='int')
    input_shape = label_dense.shape
    label_dense = label_dense.ravel()
    num_label = label_dense.shape[0]

    if not num_classes:
        num_classes = np.max(label_dense) + 1

    label_one_hot = np.zeros(shape=(num_label, num_classes), dtype=dtype)

    label_one_hot[np.arange(num_label), label_dense] = 1
    output_shape = input_shape + (num_classes,)
    label_one_hot = np.reshape(label_one_hot, output_shape)
    return label_one_hot


def train_generate(batch_size):
    """
    generator
    :param batch_size:
    :return:
    """
    pass


if __name__ == "__main__":

    # (train_images, train_label), (test_images, test_labels) = load_mnist_data(mnist_path)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # show image
    train_images = mnist.train.images
    # convert data shape
    img = train_images[10].reshape((28, 28))
    img = img.astype('float32')/255.

    plt.imshow(img)
    plt.show()

    # date nodes -> placeholder
    # create input node placeholder
    # None indicates an indefinite number
    x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 784))
    y_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 10))


    # storage nodes -> variable
    # create w b variable
    W = tf.compat.v1.Variable(initial_value=tf.zeros([784, 10]))
    b = tf.compat.v1.Variable(initial_value=tf.zeros([10]))


    # compute nodes -> operation
    # compute process
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # optimizer
    # loss function
    # tf.reduce_sum: compute the sum
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # optimize algorithm
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)


    # evaluate model
    # return False|True
    # note Flase=0 True=1
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy mean
    # tf.reduce_mean: computes the mean
    # tf.cast: cast type
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # running enviroment -> session
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist.train.next_batch(50)

            if i % 1000 == 0:
                train_accuracy =  accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                print('train accuracy {0}'.format(train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        # evaluate model
        print("test accuracy {0}".format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))




