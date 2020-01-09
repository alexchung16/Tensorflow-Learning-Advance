#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : generate_TFRecord.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/20 PM 16:05
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import cv2 as cv

original_dataset_dir = '/home/alex/Documents/datasets/dogs_vs_cat_separate'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, '100')


def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.mkdir(path)


try:
    if os.path.exists(original_dataset_dir) is False:
        print('dataset is not exist please check the path')
    else:
        if os.path.exists(tfrecord_dir) is False:
            os.mkdir(tfrecord_dir)
            print('{0} has been created'.format(tfrecord_dir))
        else:
            print('{0} has been exist'.format(tfrecord_dir))

except FileNotFoundError as e:
    print(e)


def write_tfrecord(inputs_path, outputs_record_path, shuffle=True):

    image_names, labels, classes_name = get_label_data(inputs_path)
    # use TFRecordWrite to write tfrecord
    write = tf.io.TFRecordWriter(outputs_record_path)
    for img_file, label, class_name in zip(image_names, labels, classes_name):
        # array_img = Image.open(img_file)
        # reshape_img = array_img.resize(shape_size)
        brg_img = cv.imread(img_file)
        image = cv.cvtColor(brg_img, cv.COLOR_BGR2RGB)
        img_height = image.shape[0]
        img_width = image.shape[1]
        img_depth = image.shape[2]

        img_example = image_example(label, image, img_height, img_width, img_depth, class_name.encode())
        write.write(img_example)
    write.close()


def get_label_data(data_path, classes=None, shuffle=True):
    """
    get image list and label list
    :param data_path:
    :return:
    """

    img_path = []  # save image name
    img_labels = []  # save image name
    img_names = []  # save image name

    if not classes:
        # classes name
        classes = []
        for subdir in sorted(os.listdir((data_path))):
            if os.path.isdir(os.path.join(data_path, subdir)):
                classes.append(subdir)
    num_classes = len(classes)
    class_indices = dict(zip(classes, range(num_classes)))

    for class_name in classes:
        # get image file each of class
        class_dir = os.path.join(data_path, class_name)
        image_list = os.listdir(class_dir)

        for image_name in image_list:
            img_path.append(os.path.join(class_dir, image_name))
            img_labels.append(class_indices[class_name])
            img_names.append(image_name)
    num_samples = len(img_names)

    if shuffle:
        img_path_shuffle = []
        img_labels_shuffle = []
        img_names_shuffle = []
        index_array = np.random.permutation(num_samples)

        for i, index in enumerate(index_array):
            img_path_shuffle.append(img_path[index])
            img_labels_shuffle.append(img_labels[index])
            img_names_shuffle.append(img_names[index])
        img_path = img_path_shuffle
        img_labels = img_labels_shuffle
        img_names = img_names_shuffle
    # decode label to int
    # label_encode = LabelEncoder().fit(labels)
    # labels = label_encode.transform(labels).tolist()

    return img_path, img_labels, img_names


# protocol buffer(protobuf)
# Example 是 protobuf 协议下的消息体
# 一个Example 消息体包含一系列 feature 属性
# 每个feature 是一个map (key-value)
# key 是 String类型：
# value 是 Feature 类型的消息体，取值有三种类型： BytesList， FloatList， Int64List

def _bytes_feature(value):
    """
    return bytes_list from a string / bytes
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """
    return float_list from a float / double
    :param value:
    :return:
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """
    return an int64_list from a bool/enum/int/uint.
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(label, image, filename):
    """
    create a tf.Example message to be written to a file
    :param label: label info
    :param image: image content
    :param filename: image name
    :return:
    """
    # create a dict mapping the feature name to the tf.Example compatible
    feature = {
        "label": _int64_feature(label),
        "image": _bytes_feature(image),
        "filename": _bytes_feature(filename)
    }
    # create a feature message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def image_example(label, image, img_length, img_width, img_depth, filename):
    """
    create a tf.Example message to be written to a file
    :param label: label info
    :param image: image content
    :param filename: image name
    :return:
    """

    # create a dict mapping the feature name to the tf.Example compatible
    # image_shape = tf.image.decode_jpeg(image_string).eval().shape
    feature = {
        "label": _int64_feature(label),
        "image_raw": _bytes_feature(image.tobytes()),
        "height": _int64_feature(img_length),
        "width": _int64_feature(img_width),
        "depth": _int64_feature(img_depth),
        "filename": _bytes_feature(filename)
    }
    # create a feature message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(label, image, filename):
    tf_string = tf.py_function(func=serialize_example,
                               inp=(label, image, filename),
                               Tout=tf.string)
    # the result is scalar
    return tf.reshape(tf_string, ())


def decode_message(message):
    """
    decode message from string
    :param message:
    :return:
    """
    return tf.train.Example.FromString(message)


def tfrecord_test():
    """
    test tfrecord
    :return:
    """
    # test feature type
    f0 = _bytes_feature([b'alex'])  # <class 'tensorflow.core.example.feature_pb2.Feature'>
    f1 = _bytes_feature([u'alex'.encode('utf8')])
    f2 = _float_feature([np.exp(1)])
    f3 = _int64_feature([True])
    print(type(f0))
    # serialize to binary-string
    fs = f3.SerializeToString()
    print(fs)

    # create observation dataset
    n_observation = int(1e4)
    # boolean feature
    f4 = np.random.choice([True, False], n_observation)
    # integer feature
    f5 = np.random.randint(0, 5, n_observation)
    class_str = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    # string(byte) feature
    f6 = class_str[f5]
    # string(byte) feature
    f7 = np.random.choice([b'yes', b'no'], n_observation)

    example_observation = serialize_example([True], [b'alex'], [b'yes'])
    print(example_observation)
    print(decode_message(example_observation))

    # return dataset of scalar
    feature_dataset = tf.data.Dataset.from_tensor_slices((f5, f6, f7))
    print(feature_dataset)
    # print(tf_serialize_example(f5, f6, f7))
    # apply the function to each element in dataset
    serialize_feature_dataset = feature_dataset.map(tf_serialize_example)
    print(serialize_feature_dataset)

    data_path = '/home/alex/Documents/datasets/dogs_and_cat_separate'

    # img_shape = tf.image.decode_jpeg(byte_img)
    # print(image_example(1, byte_img, b'cat'))


if __name__ == "__main__":
    image_names, labels, classes_name = get_label_data(train_path)

    record_file = os.path.join(tfrecord_dir, 'image.tfrecords')
    with tf.Session() as sess:
        write_tfrecord(inputs_path=train_path, outputs_record_path=record_file, shuffle=True)
        # show image
