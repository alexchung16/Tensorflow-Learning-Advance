#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tf_broadcast.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/21 下午2:48
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf


#-------------------numpy broadcast-------------------------
n = np.array([[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]])
m = np.array([2, 3, 3])
m = m[np.newaxis, np.newaxis, :] # [[[2, 3, 3]]]
max_m_n = np.maximum(m[..., :2], n[..., :2])  # [[[2 3], [4 5]] [[2 3],[4 5]]]
n_area = n[..., 2] * n[..., 2] # [[ 9 36], [ 9 36]]


# -------------------tensorflow broadcast---------------------
a = tf.Variable([[1,2,3], [4,5,6]], dtype=tf.int32)
pred_xywh = tf.get_variable(shape=(1, 2, 2, 3, 4), dtype=tf.float32, name="pred")
bboxes = tf.get_variable(shape=(1, 5, 4), dtype=tf.float32, name="bbox")

pred_xywh = pred_xywh[:, :, :, :, np.newaxis, :] # [1, 2, 2, 3, 1, 4]
bboxes = bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :] # [1, 1, 1, 1, 5, 4]

def bbox_iou(boxes1, boxes2):
    """

    :param boxes1: [batch_size, target_seize, target_size, 3, 1,   4]
    :param boxes2: [batch_size, 1,            1,           1, 150, 4]
    :return:
    """

    # get boxes1 area
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # [batch_size, target_seize, target_size, 3, 1]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]  # [batch_size, 1,            1,           1, 150]

    # (x, y, w, h) => (x_min, y_min, x_max, y_max)
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)  # [batch_size, target_seize, target_size, 3, 1,   4]
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)  # [batch_size, 1,            1,           1, 150, 4]

    # get inter bbox
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])  # [batch_size, target_seize, target_size, 3, 150,  2]
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:]) # # [batch_size, target_seize, target_size, 3, 150,  2]
    inter_section = tf.maximum(right_down - left_up, 0.0)

    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area  # [[batch_size, target_seize, target_size, 3, 150]

    return iou

if __name__ == "__main__":

    a_sum = tf.reduce_max(a, axis=1)

    iou = bbox_iou(pred_xywh, bboxes)
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    a_area = a[..., 2] * a[..., 2]

    init_op  = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        print(a.eval())
        print(a_sum.eval())
        print(a_area.eval())




