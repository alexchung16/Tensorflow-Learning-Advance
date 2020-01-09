#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File split_dataset.py
# @ Description :
# @ Author alexchung
# @ Time 11/12/2019 AM 11.13

import os
import math
import shutil
import numpy as np

original_dataset_dir = '/home/alex/Documents/datasets/Pascal_VOC_2012/VOCtrainval/VOCdevkit'

def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.makedirs(path)
        print('{0} has been created'.format(path))


def split_pascal(origin_path, split_rate=0.8):
    """
    split pascal dataset
    :param origin_path:
    :return:
    """
    image_path = os.path.join(origin_path, 'VOC2012/JPEGImages')
    xml_path = os.path.join(origin_path, 'VOC2012/Annotations')
    dir_path = os.path.dirname(origin_path)
    dir_name = os.path.basename(origin_path)
    image_train_path = os.path.join(dir_path, dir_name + '_train/JPEGImages')
    image_test_path = os.path.join(dir_path, dir_name + '_test/JPEGImages')
    xml_train_path = os.path.join(dir_path, dir_name + '_train/Annotations')
    xml_test_path = os.path.join(dir_path, dir_name + '_test/Annotations')
    # create path
    makedir(image_train_path)
    makedir(image_test_path)
    makedir(xml_train_path)
    makedir(xml_test_path)

    image_list = os.listdir(image_path)
    image_name = [image.split('.')[0] for image in image_list]
    image_name = np.random.permutation(image_name)
    train_image = image_name[:int(math.ceil(len(image_name) * split_rate))]
    test_image = image_name[int(math.ceil(len(image_name) * split_rate)):]

    for n, image in enumerate(train_image):
        shutil.copy(os.path.join(image_path, image+'.jpg'), os.path.join(image_train_path, image+'.jpg'))
        shutil.copy(os.path.join(xml_path, image + '.xml'), os.path.join(xml_train_path, image + '.xml'))
    print('Total of {0} data split to {1}'.format(len(train_image), os.path.join(dir_path, dir_name + '_train')))

    for n, image in enumerate(test_image):
        shutil.copy(os.path.join(image_path, image+'.jpg'), os.path.join(image_test_path, image+'.jpg'))
        shutil.copy(os.path.join(xml_path, image + '.xml'), os.path.join(xml_test_path, image + '.xml'))
    print('Total of {0} data split to {1}'.format(len(test_image), os.path.join(dir_path, dir_name + '_test')))

if __name__ == "__main__":

    image_list = os.listdir(os.path.join(original_dataset_dir, 'VOC2012/JPEGImages'))
    image_name = [image.split('.')[0] for image in image_list]

    split_pascal(original_dataset_dir, 0.8)






