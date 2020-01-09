#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File parse_raw_dataset.py
# @ Description :
# @ Author alexchung
# @ Time 10/10/2019 PM 15:35


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


DATA_PATH = '/home/alex/Documents/datasets/cifar-10-batches-py/'
BIN_DADA_PATH = '/home/alex/Documents/datasets/cifar-10-batches-bin/'

train_path = '/home/alex/Documents/datasets/dogs_cat_binary_separate/train'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def unpicpkleBin(bin_path, label_length, img_height, img_width, img_channel, label_forward=True):
    """
    unpickle binary file
    :param bin_path:
    :param label_length: label length bytes
    :param img_length: image length
    :param img_wigth: image width
    :param img_channel: image channel num
    :param label_head: label is head
    :return:
    """
    # get bin file list
    file_list = os.listdir(bin_path)
    bin_list = [file for file in file_list if os.path.splitext(file)[1] == '.bin']
    image_vec_bytes = label_length + img_height * img_width * img_channel
    image_length = img_height * img_width * img_channel

    labels = np.zeros((0, label_length), dtype=tf.uint8)
    images = np.zeros((0, image_length), dtype=tf.uint8)
    for bin_file in bin_list:
        with open(os.path.join(bin_path, bin_file), 'rb') as f:
            bin_data = f.read()
        data = np.frombuffer(bin_data, dtype=np.uint8)
        data = data.reshape(-1, image_vec_bytes)
        # save label and image data
        label = None
        image = None
        if label_forward:
            label_image = np.hsplit(data, [label_length])
            label = label_image[0]
            image = label_image[1]
        else:
            image_label = np.hsplit(data, [image_length])
            label = image_label[1]
            image = image_label[0]
        # stack array
        labels = np.vstack((labels, label))
        images = np.vstack((images, image))

    return labels, images


def convertAndShowImage(img_vector, img_height, img_width, img_channel):
    """
    convert vector shape to [img_length, img_width, img_channel]
    show image
    :param img_vector:
    :param img_length:
    :param img_width:
    :param img_channel:
    :return:
    """
    # img = np.reshape(img, [-1, 3, 32, 32])
    # r = img[0][0]
    # g = img[0][1]
    # b = img[0][2]
    # ir = Image.fromarray(r)
    # ig = Image.fromarray(g)
    # ib = Image.fromarray(b)
    # img = Image.merge("RGB", (ir, ig, ib))
    # img = Image.merge('RGB', (r_channel, g_channel, b_channel))
    #  Image._show(img)

    # r_channel = img[0: 1024].reshape(32, 32)
    # g_channel = img[1024: 2048].reshape(32, 32)
    # b_channel = img[2048:].reshape(32, 32)
    # img = np.dstack((r_channel, g_channel, b_channel))
    img = np.reshape(img_vector, (img_channel, img_height, img_width))
    img = img.transpose((1, 2, 0))
    img = img * 1. / 255.
    plt.imshow(img)
    plt.show()


def readRecord(file_list, label_length, img_height, img_width, img_channel):
    """
    use tensorflow read binary file
    :param file_list:
    :param label_length:
    :param img_length:
    :param img_width:
    :param img_channel:
    :return:
    """
    # compute record bytes
    images_bytes = img_height * img_width * img_channel
    record_bytes_length = label_length + img_height * img_width * img_channel
    # create a queue that yproduce the file name to read
    filename_queue = tf.train.string_input_producer(file_list)
    # read a record
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes_length)
    # getting filenames from filename_queue
    key, recording_string = reader.read(queue=filename_queue)
    # convert from a string to a vector of uint8
    record_bytes = tf.decode_raw(input_bytes=recording_string, out_type=tf.uint8)
    # decode label
    # the first bytes represent the label, which convert from uint8 to uint32
    labels = tf.cast(tf.slice(input_=record_bytes, begin=[0], size=[label_length]), dtype=tf.uint32)
    # decode image
    # reshape image from [img_length * img_width * img_channel] to [img_length, img_width, img_channel]
    depth_major = tf.reshape(tf.slice(input_=record_bytes, begin=[label_length], size=[images_bytes]),
                                   shape=[img_channel, img_width, img_height])
    raw_images = tf.transpose(a=depth_major, perm=[1, 2, 0])
    # normalize image
    images = tf.image.per_image_standardization(image=raw_images)

    return labels, images

def readMetaData(meta_path):
    """
    read meta data
    :param meta_path:
    :return:
    """
    classes = []
    with open(meta_path, 'r') as fr:
        all_lines = fr.readlines()
        for i, class_name in enumerate(all_lines):
            classes.append(class_name.strip('\n'))
        fr.close()
    return classes


if __name__ == "__main__":
    LABEL_LENGTH = 1
    IMAGE_LENGTH = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNEL = 3

    data_path_1 = DATA_PATH + 'data_batch_1'
    # bin_data_path = BIN_DADA_PATH + 'data_batch_1.bin'
    bin_data_path = os.path.join(train_path, 'image_record.bin')
    meta_data_path = os.path.join(train_path, 'meta.txt')

    data = unpickle(data_path_1)
    bin_label, bin_image = unpicpkleBin(train_path, LABEL_LENGTH, IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_CHANNEL)
    # print(data.keys())  # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    # img1 = data[b'data'][12]
    bin_img1 = bin_image[1]
    # convertAndShowImage(img1,LABEL_LENGTH, IMAGE_LENGTH, IMAGE_WIDTH)
    convertAndShowImage(bin_img1, IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_CHANNEL)
    # print(bin_label)
    print(bin_image.shape)

    # test readRecord function
    # # file name list
    # filenames = [bin_data_path]
    # with tf.Session() as sess:
    #     labels, images = readRecord(filenames,  LABEL_LENGTH, IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_CHANNEL)
    #     image = sess.run(tf.shape(images))
    #     print(image)

    file_list = os.listdir(train_path)
    for file in file_list:
        print(os.path.splitext(file))


