#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_conv2d_padding.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/17 上午11:07
# @ Software   : PyCharm
#-------------------------------------------------------
import numpy as np
import tensorflow as tf


def conv2d_custom(input, filter=None, strides=None, padding=None, data_format="NHWC", name=None):
    """
    custom conv2d to evaluate padding operation
    :param input:
    :param filter:
    :param strides:
    :param padding:
    :param data_format:
    :param name:
    :return:
    """
    net = None
    if padding == 'VALID':
        net = tf.nn.conv2d(input=input, filter=filter, strides=strides, data_format=data_format, name=name)
    elif padding == "SAME":
        input_shape = list(map(int, list(input.get_shape())))
        filter_shape = list(map(int, list(filter.get_shape())))
        # ------------------------------padding part-----------------------------------
        # step 1 get outputs shape

        height_size = int(np.ceil(input_shape[1]/ strides[1]))
        width_size = int(np.ceil(input_shape[2] / strides[2]))

        # step 2 get padding size
        num_height_padding = (height_size - 1) * strides[1] + filter_shape[0] - input_shape[1]
        num_width_padding = (width_size -1) * strides[2] + filter_shape[1] - input_shape[2]

        height_top = int(np.floor(num_height_padding / 2))
        height_bottom = num_height_padding - height_top
        width_left = int(np.floor(num_width_padding / 2))
        width_right = num_width_padding - width_left

        if data_format == "NHWC":
            padding = [[0, 0], [height_top, height_bottom], [width_left, width_right], [0, 0]]
        elif data_format == "NCHW":
            padding = [[0, 0], [0, 0], [height_top, height_bottom], [width_left, width_right]]

        # step 3  execute padding operation
        padding_input = tf.pad(tensor=input, paddings=padding)
        # ------------------------------ VALID convolution part----------------------------
        # step 4 execute convolution operation
        net = tf.nn.conv2d(input=padding_input, filter=filter, strides=strides, data_format=data_format,
                           padding="VALID", name=name)

    return net


def main():
    # ++++++++++++++++++++++++++++++++++++++config and data part+++++++++++++++++++++++++++++
    BATCH_SIZE = 6
    IMAGE_HEIGHT = 5
    IMAGE_WIDTH = 5
    CHANNELS = 3
    NUM_OUTPUTS = 6
    STRIDE = [1, 2, 2, 1]
    tf.random.set_random_seed(0)
    image_batch = tf.Variable(tf.random_uniform(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), minval=0, maxval=225,
                                          dtype=tf.float32))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #+++++++++++++++++++++++++++++++++ custom SAME mode of ood case++++++++++++++++++++++++++++++++++++
    filters_3x3 = tf.Variable(initial_value=tf.random_uniform(shape=[3, 3, CHANNELS, NUM_OUTPUTS]))
    with tf.variable_scope("part_1"):
        output_same_3x3 = tf.nn.conv2d(input=image_batch, filter=filters_3x3, strides=STRIDE, padding='SAME', name='same_3x3')
        output_valid_3x3 = tf.nn.conv2d(input=image_batch, filter=filters_3x3, strides=STRIDE, padding='VALID', name='valid_3x3')

        # custom SAME mode of convolution
        #  sample 1:  filter size is 3
        # input_size = 5, filter_size = 3, stride = 2
        # => output_size = np.ceil(5 / 2) = 3
        # => num_padding = (3 -1) * 2 + 3 - 5 = 2
        # according to the padding rule (where the num_padding is even)
        # => height_top = np.floor(2 / 2) = 1
        # => height_bottom = 2 -1  = 1
        # => width_left = np.floor(2 / 2) = 1
        # => width_right = 2 - 1 = 1
        padding_batch_3x3 = tf.pad(tensor=image_batch, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        # step 2 execute convolution operation
        output_same_3x3_custom = tf.nn.conv2d(input=padding_batch_3x3, filter=filters_3x3, strides=STRIDE, padding='VALID',
                                               name='custom_same_3x3')

    init_op_1 = tf.group(tf.local_variables_initializer(),
                       tf.global_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op_1)
        print("custom SAME mode of ood case:")
        # paddings == 'SAME'
        assert output_same_3x3.shape == (BATCH_SIZE,
                                         np.ceil(IMAGE_HEIGHT / STRIDE[1]),
                                         np.ceil(IMAGE_WIDTH / STRIDE[2]),
                                         NUM_OUTPUTS)
        print(output_same_3x3.shape)  # (6, 3, 3, 6)

        # paddings == 'VALID'
        # output_size = np.ceil((input_size - filter_size + 1) / stride)
        # output_size = np.ceil((5 -3 + 1) / 2) = 2
        assert output_valid_3x3.shape == (BATCH_SIZE,
                                         np.ceil((IMAGE_HEIGHT - 3 + 1)/ STRIDE[1]),
                                         np.ceil((IMAGE_WIDTH - 3 + 1) / STRIDE[2]),
                                         NUM_OUTPUTS)
        print(output_valid_3x3.shape)  # (6, 2, 2, 6)

        assert output_same_3x3_custom.shape == (BATCH_SIZE,
                                                np.ceil(IMAGE_HEIGHT / STRIDE[1]),
                                                np.ceil(IMAGE_WIDTH / STRIDE[2]),
                                                NUM_OUTPUTS)
        # the custom operation result is equal to office interface
        assert (sess.run(output_same_3x3) == sess.run(output_same_3x3_custom)).all()
        print(output_same_3x3_custom.shape) # (6, 2, 2, 6)

    # +++++++++++++++++++++++++++++++++ custom SAME mode of even case++++++++++++++++++++++++++++++++++++
    with tf.variable_scope("part_2"):
        filters_4x4 = tf.Variable(initial_value=tf.random_uniform(shape=[4, 4, CHANNELS, NUM_OUTPUTS]))

        output_same_4x4 = tf.nn.conv2d(input=image_batch, filter=filters_4x4, strides=STRIDE, padding='SAME',
                                       name='same_4x4')
        output_valid_4x4 = tf.nn.conv2d(input=image_batch, filter=filters_4x4, strides=STRIDE, padding='VALID',
                                        name='valid_4x4')
        # sample 2: filter size is 4
        # input_size = 5, filter_size = 4, stride = 2
        # => output_size = np.ceil(5 / 2) = 3
        # => num_padding = (3 -1) * 2 + 4 - 5 = 3
        # according to the padding rule(where the num_padding is odd)
        # => height_top = np.floor(3 / 2) = 1
        # => height_bottom = 3 - 1 = 2
        # => width_left = np.floor(3 / 2) = 1
        # => width_right = 3 - 2  = 2

        padding_batch_4x4 = tf.pad(tensor=image_batch, paddings=[[0, 0], [1, 2], [1, 2], [0, 0]])
        output_same_4x4_custom = tf.nn.conv2d(input=padding_batch_4x4, filter=filters_4x4, strides=STRIDE, padding='VALID',
                                              name='custom_same_4x4')

    init_op_2 = tf.group(tf.local_variables_initializer(),
                         tf.global_variables_initializer())
    with tf.Session(config=config) as sess:
        # +++++++++++++++++++++++++++++++++ custom SAME mode++++++++++++++++++++++++++++++++++++
        sess.run(init_op_2)
        print("custom SAME mode of even case:")
        assert output_valid_4x4.shape == (BATCH_SIZE,
                                          np.ceil((IMAGE_HEIGHT - 4 + 1) / STRIDE[1]),
                                          np.ceil((IMAGE_WIDTH - 4 + 1) / STRIDE[2]),
                                          NUM_OUTPUTS)
        print(output_valid_4x4.shape)  # (6, 1, 1, 6)

        assert output_same_4x4_custom.shape == (BATCH_SIZE,
                                                np.ceil(IMAGE_HEIGHT / STRIDE[1]),
                                                np.ceil(IMAGE_WIDTH / STRIDE[2]),
                                                NUM_OUTPUTS)
        # the custom operation result is equal to office interface
        assert (sess.run(output_same_4x4) == sess.run(output_same_4x4_custom)).all()
        print(output_same_4x4_custom.shape)  # (6, 3, 3, 6)

    #+++++++++++++++++++++++++++++++++test custom conv2d module++++++++++++++++++++++++++++++++++
    with tf.variable_scope("part_3"):
        custom_same_3x3 = conv2d_custom(input=image_batch, filter=filters_3x3, strides=STRIDE, padding='SAME',
                                        name = "custom_same_3x3")

        custom_same_4x4 = conv2d_custom(input=image_batch, filter=filters_4x4, strides=STRIDE, padding='SAME',
                                        name = "custom_same_4x4")

    init_op_3 = tf.group(tf.local_variables_initializer(),
                         tf.global_variables_initializer())
    with tf.Session(config=config) as sess:
        # +++++++++++++++++++++++++++++++++ custom SAME mode++++++++++++++++++++++++++++++++++++
        sess.run(init_op_3)
        print("test custom conv2d module:")
        assert custom_same_3x3.shape == (BATCH_SIZE,
                                         np.ceil(IMAGE_HEIGHT / STRIDE[1]),
                                         np.ceil(IMAGE_WIDTH / STRIDE[2]),
                                         NUM_OUTPUTS)
        # the custom operation result is equal to office interface
        assert (sess.run(output_same_3x3) == sess.run(custom_same_3x3)).all()
        print(custom_same_3x3.shape)  # (6, 3, 3, 6)

        assert custom_same_4x4.shape == (BATCH_SIZE,
                                         np.ceil(IMAGE_HEIGHT / STRIDE[1]),
                                         np.ceil(IMAGE_WIDTH / STRIDE[2]),
                                         NUM_OUTPUTS)
        # the custom operation result is equal to office interface
        assert (sess.run(output_same_4x4) == sess.run(custom_same_4x4)).all()
        print(custom_same_4x4.shape)  # (6, 3, 3, 6)


if __name__ == "__main__":
    main()

#+++++++++++++++++++++++++++how to get padding size++++++++++++++++++++++++++++++++++++++
# ------------------------------rule elaborate part----------------------------------------
# reference formula output_size = np.floor(input_size + num_padding - filter_size) / stride + 1
# => num_padding  = (output_size - 1) * stride + filter_size - input_size
# according to the padding rule of conv2d interface
# => height_top = np.floor(num_padding / 2)
# => height_bottom = num_padding - height_top
# => width_left = np.floor(num_padding / 2)
# => width_right = num_padding - width_left
# => paddings = [[0, 0], [height_top, height_bottom], [width_left, width_right], [0, 0]]