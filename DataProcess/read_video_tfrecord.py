#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : read_video_tfrecord.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/26 下午6:45
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.python_io import tf_record_iterator

# original_dataset_dir = '/home/alex/Documents/datasets/dogs_vs_cat_separate'
original_dataset_dir = '/home/alex/Documents/dataset/bike_raft'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, 'test')


def parse_example(serialized_sample, clip_size, input_shape, class_depth):
    """
    parse tensor
    :param image_sample:
    :return:
    """

    # construct feature description
    image_feature_description ={

        "rgb_video": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "flow_video": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "height": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "width": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "rgb_depth": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "flow_depth": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "rgb_frames": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "flow_frames": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }
    feature = tf.io.parse_single_example(serialized=serialized_sample, features=image_feature_description)

    # parse feature
    rgb_video = tf.decode_raw(feature['rgb_video'], tf.uint8)
    flow_video = tf.decode_raw(feature['flow_video'], tf.float32)
    # shape = tf.cast(feature['shape'], tf.int32)
    height = tf.cast(feature['height'], tf.int32)
    width = tf.cast(feature['width'], tf.int32)
    depth = tf.cast(feature['depth'], tf.int32)
    rgb_depth = tf.cast(feature['rgb_depth'], tf.int32)
    flow_depth = tf.cast(feature['flow_depth'], tf.int32)
    rgb_frames = tf.cast(feature['rgb_frames'], tf.int32)
    flow_frames = tf.cast(feature['flow_frames'], tf.int32)
    label = tf.cast(feature['label'], tf.int32)

    rgb_video = tf.reshape(rgb_video, [rgb_frames, height, width, rgb_depth])
    flow_video = tf.reshape(flow_video, [rgb_frames, height, width, flow_depth])

    filename = tf.cast(feature['filename'], tf.string)
    # resize image shape
    # random crop image
    # before use shuffle_batch, use random_crop to make image shape to special size
    # first step enlarge image size
    # second step dataset operation

    # image augmentation
    rgb_video = augmentation_video(input_video=rgb_video, image_shape=input_shape)
    flow_video = augmentation_video()
    # onehot label
    label = tf.one_hot(indices=label, depth=class_depth)

    return rgb_video, label, filename


def augmentation_video(input_video, ):
    # enlarge image to same size

    try:
      pass
    except Exception as e:
        print(e)


def dataset_tfrecord(record_file, input_shape, class_depth, epoch=5, batch_size=10, shuffle=True):
    """
    construct iterator to read image
    :param record_file:
    :return:
    """
    record_list = []
    # check record file format
    if os.path.isfile(record_file):
        record_list = [record_file]
    else:
        for filename in os.listdir(record_file):
            record_list.append(os.path.join(record_file, filename))
    # # use dataset read record file
    raw_img_dataset = tf.data.TFRecordDataset(record_list)
    # execute parse function to get dataset
    # This transformation applies map_func to each element of this dataset,
    # and returns a new dataset containing the transformed elements, in the
    # same order as they appeared in the input.
    # when parse_example has only one parameter (office recommend)
    # parse_img_dataset = raw_img_dataset.map(parse_example)
    # when parse_example has more than one parameter which used to process data
    parse_img_dataset = raw_img_dataset.map(lambda series_record:
                                            parse_example(series_record, input_shape, class_depth))
    # get dataset batch
    if shuffle:
        shuffle_batch_dataset = parse_img_dataset.shuffle(buffer_size=batch_size*4).repeat(epoch).batch(batch_size=batch_size)
    else:
        shuffle_batch_dataset = parse_img_dataset.repeat(epoch).batch(batch_size=batch_size)
    # make dataset iterator
    image, label, filename = shuffle_batch_dataset.make_one_shot_iterator().get_next()

    # image = augmentation_image(input_image=image, image_shape=input_shape)
    # # onehot label
    # label = tf.one_hot(indices=label, depth=class_depth)

    return image, label, filename


def get_num_samples(record_dir):
    """
    get tfrecord numbers
    :param record_file:
    :return:
    """

    record_list = []
    # check record file format

    for filename in os.listdir(record_dir):
        record_list.append(os.path.join(record_dir, filename))

    num_samples = 0
    for record_file in record_list:
        for record in tf_record_iterator(record_file):
            num_samples += 1
    return num_samples

if __name__ == "__main__":
    record_file = os.path.join(tfrecord_dir, 'train')
    image_batch, label_batch, filename = dataset_tfrecord(record_file=record_file, input_shape=[224, 224, 3],
                                                          class_depth=5)
    # create local and global variables initializer group
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)

        num_samples = get_num_samples(record_file)
        print('all sample size is {0}'.format(num_samples))
        # create Coordinator to manage the life period of multiple thread
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph to execute input queue operation
        # the step contain two operation:filename to filename queue and sample to sample queue
        threads = tf.train.start_queue_runners(coord=coord)
        print('threads: {0}'.format(threads))
        try:
            if not coord.should_stop():
                image_feed, label_feed = sess.run([image_batch, label_batch])
                plt.imshow(image_feed[0])
                plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()

        # waiting all threads safely exit
        coord.join(threads)
        sess.close()
