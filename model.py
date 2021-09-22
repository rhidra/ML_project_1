import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import data

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_hat = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
        y_hat = K.clip(y_hat, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_hat) * weights
        loss = - K.sum(loss) / K.cast(K.shape(y_true)[0] * K.shape(y_true)[1] * K.shape(y_true)[2], 'float')
        return loss

    return loss

def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_model():
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(data.IMG_SIZE, data.IMG_SIZE, 3))
    for layer in resnet.layers:
        layer.trainable = False
        if layer.name == 'activation_71':
            break

    end_stage1 = resnet.get_layer('activation_1').output
    end_stage2 = resnet.get_layer('activation_10').output
    X = resnet.get_layer('activation_22').output

    # Out of ResNet: [32, 32, 512]
    X = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = UpSampling2D(size=2)(X) # Out: [64, 64, 512]
    X = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = concatenate([end_stage2, X], axis=3)
    X = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = UpSampling2D(size=2)(X) # Out: [128, 128, 256]
    X = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = concatenate([end_stage1, X], axis=3)
    X = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = UpSampling2D(size=2)(X) # Out: [256, 256, 64]
    X = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = Conv2D(21, 3, activation='softmax', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=resnet.inputs, outputs=X)

    return model

def get_model_without_concat():
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(data.IMG_SIZE, data.IMG_SIZE, 3))
    for layer in resnet.layers:
        layer.trainable = False

    X = resnet.get_layer('activation_22').output

    # Out of ResNet: [64, 64, 512]
    X = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = UpSampling2D(size=2)(X) # Out: [128, 128, 512]
    X = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = UpSampling2D(size=2)(X) # Out: [256, 256, 256]
    X = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = UpSampling2D(size=2)(X) # Out: [512, 512, 256]
    X = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = Conv2D(21, 3, activation='softmax', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=resnet.inputs, outputs=X)

    return model


def predict_dataset(m, test_gen, testing_size, batch_size=32):
    y_true = np.zeros((testing_size, data.IMG_SIZE, data.IMG_SIZE))
    y_pred = np.zeros((testing_size, data.IMG_SIZE, data.IMG_SIZE))
    for i in range(testing_size // batch_size):
        x, y = next(test_gen)
        pred = m.predict(x)
        y_true[i * batch_size:(i+1) * batch_size] = np.argmax(y, axis=3)
        y_pred[i * batch_size:(i+1) * batch_size] = np.argmax(pred, axis=3)
    return y_true, y_pred
