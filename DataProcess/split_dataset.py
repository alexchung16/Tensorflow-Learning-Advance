#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File split_dataset.py
# @ Description :
# @ Author alexchung
# @ Time 11/12/2019 AM 11.13

import os
import math
import json
import shutil
import numpy as np
from collections import defaultdict

# original_dataset_dir = '/home/alex/Documents/datasets/Pascal_VOC_2012/VOCtrainval/VOCdevkit'

data_type = 'val2017'
dataset_dir = '/home/alex/Documents/dataset/COCO_2017'

img_dir = os.path.join(dataset_dir, data_type)
instance_dir = '{0}/annotations_trainval2017/annotations/instances_{1}.json'.format(dataset_dir, data_type)
keypoint_dir = '{0}/annotations_trainval2017/annotations/person_keypoints_{1}.json'.format(dataset_dir, data_type)
caption_dir = '{0}/annotations_trainval2017/annotations/captions_{1}.json'.format(dataset_dir, data_type)

sub_coco = '/home/alex/Documents/dataset/COCO_2017/sub_coco'
sub_annotations = os.path.join(sub_coco, 'Annotations')
sub_images = os.path.join(sub_coco, 'Images')

def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.makedirs(path)
        print('{0} has been created'.format(path))

#++++++++++++++++++++++++++++++++++++++++split pascal++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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


#++++++++++++++++++++++++++++++++++++++++split coco++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def split_coco(imgs_path, annotaions_path, dst_dir, per_cate_base_num=2):
    """

    :param origin_path:
    :param split_ratio:
    :return:
    """

    dataset = json.load(open(annotaions_path, 'r'))

    sub_annotations_path = os.path.join(dst_dir, 'Annotations')
    sub_img_path = os.path.join(dst_dir, 'Images')


    anns, cats, imgs, img_anns, cate_imgs = create_index(dataset)

    img_id_list = get_img_per_categorise(cate_imgs, per_cate_base_num)

    img_name_list = []
    for i, img_id in enumerate(img_id_list):
        img_name_list.append('0' * (12 - len(str(img_id))) + '{0}.jpg'.format(img_id))

    #----------------------------write annotaion info-----------------------------------
    images_list, annotations_list = get_images_annotaion_info(img_id_list, imgs, img_anns)
    new_dataset = defaultdict(list)
    new_dataset['info'] = dataset['info']
    new_dataset['licenses'] = dataset['licenses']
    new_dataset['images'] = images_list
    new_dataset['annotations'] = annotations_list
    new_dataset['categories'] = dataset['categories']

    makedir(sub_annotations_path)
    json_path = os.path.join(sub_annotations_path, 'instances.json')
    with open(json_path, 'w') as fw:
        json.dump(new_dataset, fw)
    print('Successful write the number of {0} annotations respect to {1} images to {2}'.
          format(len(new_dataset['annotations']), len(new_dataset['images']), json_path))

    #---------------------------------remove image---------------------------------------
    makedir(sub_img_path)
    for i, img_name in enumerate(img_name_list):
        shutil.copy(os.path.join(imgs_path, img_name), os.path.join(sub_img_path, img_name))
    print('Successful copy the number of {0} images to {1}'.format(len(img_name_list), sub_img_path))


def create_index(dataset):
    """
    create index
    :param dataset:
    :return:
    """
    print('creating index...')
    anns, cats, imgs = {}, {}, {}
    img_anns, cate_imgs = defaultdict(list), defaultdict(list)
    if 'annotations' in dataset:
        for ann in dataset['annotations']:
            img_anns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in dataset:
        for img in dataset['images']:
            imgs[img['id']] = img

    if 'categories' in dataset:
        for cat in dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in dataset and 'categories' in dataset:
        for ann in dataset['annotations']:
            cate_imgs[ann['category_id']].append(ann['image_id'])
    print('index created!')

    return  anns, cats, imgs, img_anns, cate_imgs


    # img_list = []
    # for i, img_name in enumerate(img_name):
    #     img_list.append(int(img_name.replace('0', '').split('.')[0]))

def get_img_per_categorise(cate_imgs, per_cate_base_num=2):
    """
    get image id according to per categorise has equal num
    :param cate_img:
    :param pre_cate_base_num:
    :return:
    """
    base_num = per_cate_base_num
    img_id_list = []
    for cate_id, imgs_id in cate_imgs.items():
        for img_id in imgs_id:
            if base_num != 0:
                if img_id not in img_id_list:
                    img_id_list.append(img_id)
                    base_num -= 1
            else:
                base_num = per_cate_base_num
                break

    return img_id_list

def get_images_annotaion_info(img_id_list, imgs_raw, img_anns_raw):
    """

    :param img_id_list:
    :param imgs_raw:
    :param img_anns_raw:
    :return:
    """
    images_list = []
    annotations_list = []
    annotation_index = 0
    for img_index, img_id in enumerate(img_id_list):
        img_annotations = img_anns_raw[img_id]
        img_info = imgs_raw[img_id]
        img_info['id'] = img_index
        if len(img_annotations) == 0:
            continue
        else:
            for i, annotation in enumerate(img_annotations):
                annotation['id'] = annotation_index
                annotations_list.append(annotation)
                annotation_index += 1
        images_list.append(img_info)

    return images_list, annotations_list



if __name__ == "__main__":

    # image_list = os.listdir(os.path.join(original_dataset_dir, 'VOC2012/JPEGImages'))
    # image_name = [image.split('.')[0] for image in image_list]
    #
    # split_pascal(original_dataset_dir, 0.8)

    image_list = os.listdir(img_dir)
    print(len(image_list))

    dataset = json.load(open(instance_dir, 'r'))
    print(dataset.keys())
    print(len(dataset['images']))
    annotations = dataset['annotations']
    print(len(annotations))

    split_coco(img_dir, instance_dir, sub_coco)

































