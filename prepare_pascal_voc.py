from os import path
from shutil import copy
import math

"""
Helps to prepare the PASCAL VOC dataset in the correct folders.
This file needs to be run only once.
"""

ROOT = path.abspath('.')

IMAGES_TRAIN = path.join(ROOT, 'VOC2012/ImageSets/Segmentation/train.txt')
IMAGES_TRAIN_VAL = path.join(ROOT, 'VOC2012/ImageSets/Segmentation/trainval.txt')
IMAGES_VAL = path.join(ROOT, 'VOC2012/ImageSets/Segmentation/val.txt')
IMAGES_TEST = path.join(ROOT, 'VOC2012/ImageSets/Segmentation/test.txt')

SEG_CLASS = path.join(ROOT, 'VOC2012/SegmentationClass/')
SEG_INST = path.join(ROOT, 'VOC2012/SegmentationObject/')
IMG = path.join(ROOT, 'VOC2012/JPEGImages/')

IMAGES_TRAIN_DST = path.join(ROOT, 'dataset/images_train')
ANNOTATIONS_TRAIN_DST = path.join(ROOT, 'dataset/annotations_train/')
IMAGES_TEST_DST = path.join(ROOT, 'dataset/images_test/')
ANNOTATIONS_TEST_DST = path.join(ROOT, 'dataset/annotations_test/')

# Copy training/val images
with open(IMAGES_TRAIN_VAL, 'r') as file:
    names = file.read().rstrip('\n').split('\n')
names = filter(lambda s: bool(s), names)

for i, name in enumerate(names):
    img = path.join(IMG, name + '.jpg')
    segInst = path.join(SEG_INST, name + '.png')
    segClass = path.join(SEG_CLASS, name + '.png')

    copy(img, path.join(IMAGES_TRAIN_DST, name + '.jpg'))
    copy(segClass, path.join(ANNOTATIONS_TRAIN_DST, name + '.png'))


# Copy testing images
with open(IMAGES_TEST, 'r') as file:
    names = file.read().rstrip('\n').split('\n')
names = filter(lambda s: bool(s), names)

for i, name in enumerate(names):
    img = path.join(IMG, name + '.jpg')
    segInst = path.join(SEG_INST, name + '.png')
    segClass = path.join(SEG_CLASS, name + '.png')

    copy(img, path.join(IMAGES_TEST_DST, name + '.jpg'))
    copy(segClass, path.join(ANNOTATIONS_TEST_DST, name + '.png'))
