#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File tf_image.py
# @ Description : ConvNerwork
# @ Author alexchung
# @ Time 1/12/2019 AM 11:30

import numpy as np
import tensorflow as tf
import cv2 as cv

image_path = '../Data/sunflowers.jpg'

if __name__ == "__main__":

    # --------------------tf.image.convert_image_dtype--------------------------------------------
    image = cv.imread(filename=image_path)
    cv.imshow(winname='image', mat=image)

    distort_image = cv.resize(image, dsize=(224, 224))
    custom_distort_image = tf.multiply(tf.cast(distort_image, dtype=tf.float32), 1./255)
    distort_image = tf.image.convert_image_dtype(distort_image, dtype=tf.float32)
    print(distort_image.dtype.max)

    # -------------------tf.image.non_max_suppression--------------------------------------------------------
    boxes = tf.Variable(initial_value=[[1,2, 4, 5], [1, 3, 4, 4], [2, 2, 3, 5], [3, 2 , 6, 4]], dtype=tf.float32)
    scores = tf.Variable(initial_value=[0.6, 0.9, 0.4, 0.8], dtype=tf.float32)
    select_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores, iou_threshold=0.5, max_output_size=3)
    select_boxes = tf.gather(params=boxes, indices=select_indices)

    # -----------------------------------tf.images.crop_and_resize-------------------------------------------------------
    image_feature = tf.expand_dims(tf.convert_to_tensor(image), axis=0)
    boxes = [[0.2, 0.2, 0.5, 0.6],
             [0.1, 0.3, 0.6, 0.8],
             [0.2, 0.4, 0.6, 0.5]]
    box_indices = tf.zeros(shape=[tf.shape(boxes)[0], ], dtype=tf.int32)
    crop_img = tf.image.crop_and_resize(image=image_feature, boxes=boxes, box_ind=box_indices, crop_size=[112, 112])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # assert distort_image.eval() == custom_distort_image.eval()

        assert (distort_image.eval() == custom_distort_image.eval()).all()
        print(custom_distort_image.eval())
        print('------------------------------------')
        print(distort_image.eval())

        print('------------------------------------')
        print(sess.run(select_indices))
        print(sess.run(select_boxes))

        print('------------------------------------')
        print(box_indices.eval())
        print(tf.shape(crop_img))

        crop_img_1, crop_img_2, crop_img_3 = crop_img.eval()

        cv.imshow("crop image 1", np.array(crop_img_1, np.uint8))
        cv.imshow("crop image 2", np.array(crop_img_2, np.uint8))
        cv.imshow("crop image 3", np.array(crop_img_3, np.uint8))
        cv.waitKey(0)


