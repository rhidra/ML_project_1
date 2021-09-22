import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.optimizers import Adam

import data
import model

batch_size = 4
training_size = 2913
testing_size = 210

train_gen = data.data_gen('dataset/images_train', 'dataset/annotations_train', batch_size=batch_size, one_hot=True)
test_gen = data.data_gen('dataset/images_test', 'dataset/annotations_test', batch_size=batch_size, one_hot=True)

model = model.get_model()

results = model.fit_generator(train_gen, epochs=200, steps_per_epoch=training_size // batch_size, verbose=1,callbacks=[ModelCheckpoint(filepath='model.{epoch:02d}.h5')])

print('Results', results)
