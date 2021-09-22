from os import path
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import keras
import seaborn as sns
import sklearn.metrics

CLASS_MAP = { # (R, G, B, classID)
    'aeroplane': (128, 0, 0, 1),
    'bicycle': (0, 128, 0, 2),
    'bird': (128, 128, 0, 3),
    'boat': (0, 0, 128, 4),
    'bottle':(128, 0, 128, 5),
    'bus': (0, 128, 128, 6),
    'car': (128, 128, 128, 7),
    'cat':(64, 0, 0, 8),
    'chair': (192, 0, 0, 9),
    'cow':(64, 128, 0, 10),
    'dinningtable': (192, 128, 0, 11),
    'dog': (64, 0, 128, 12),
    'horse': (192, 0, 128, 13),
    'motorbike': (64, 128, 128, 14),
    'person': (192, 128, 128, 15),
    'pottedplant': (0, 64, 0, 16),
    'sheep': (128, 64, 0, 17),
    'sofa': (0, 192, 0, 18),
    'train': (128, 192, 0, 19),
    'tvmonitor': (0, 64, 128, 20),
}

IMG_SIZE = 256

def data_gen(img_folder='dataset/images_train', mask_folder='dataset/annotations_train', batch_size=32, one_hot=False):
    c = 0
    files = os.listdir(img_folder)
    #random.shuffle(files)

    while (True):
        img = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3)).astype('float')
        mask = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, len(CLASS_MAP) + 1 if one_hot else 1)).astype('float')

        for i in range(c, c + batch_size):
            train_img = load_img(os.path.join(img_folder, files[i]), target_size=(IMG_SIZE, IMG_SIZE))
            train_img = np.array(train_img)

            img[i - c] = train_img

            train_mask = load_img(os.path.join(mask_folder, files[i][:-4] + '.png'), target_size=(IMG_SIZE, IMG_SIZE))
            train_mask = np.array(train_mask)
            train_mask = mask_rgb_to_class(train_mask)
            train_mask = train_mask.reshape(IMG_SIZE, IMG_SIZE, 1)

            if one_hot:
                train_mask = keras.utils.to_categorical(train_mask, num_classes=len(CLASS_MAP) + 1)

            mask[i - c] = train_mask

        c += batch_size
        if c + batch_size >= len(files):
            c = 0
            random.shuffle(files)
        img = keras.applications.resnet50.preprocess_input(img)
        yield img, mask


def mask_rgb_to_class(mask):
    out = np.zeros((mask.shape[0], mask.shape[1], 1))
    for _, c in CLASS_MAP.items():
        out[np.where((mask == [c[0], c[1], c[2]]).all(axis=2))] = c[3]
    return out

def mask_class_to_rgb(mask):
    out = np.zeros((mask.shape[0], mask.shape[1], 3))
    for _, c in CLASS_MAP.items():
        out[np.where(mask == c[3])] = [c[0], c[1], c[2]]
    return out


def display_result(x, y, y_hat, i):
    classes = np.argmax(y_hat[i], axis=2)
    t = np.unique(classes.reshape(-1), return_counts=True)
    for c, n in zip(t[0][1:], t[1][1:]):
        print('{:13} {:04.2f}%'.format(list(filter(lambda x: x[1][3]==c, CLASS_MAP.items()))[0][0] + ':', 100*n/(classes.size-t[1][0])))
    mask_pred = mask_class_to_rgb(classes)
    mask = mask_class_to_rgb(np.argmax(y[i], axis=2))
    plt.subplot(131)
    plt.imshow(x[i])
    plt.title('Image')
    plt.subplot(132)
    plt.imshow(mask)
    plt.title('Expected')
    plt.subplot(133)
    plt.imshow(mask_pred)
    plt.title('Predicted')


def confusion_matrix(y_true, y_pred):
    cm = sklearn.metrics.confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
    plt.figure(figsize=(16,10))
    sns.heatmap(cm + 1, annot=True, fmt="d", norm=LogNorm())
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
