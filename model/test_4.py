#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:02:11 2018

@author: changetest
"""


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ['IMAGE_DIM_ORDERING'] = 'tf'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
tf.Session(config=config)

import pandas as pd
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import cPickle as pickle
import time
import numpy as np
import h5py
import sys
from keras import backend as K
from keras.datasets import mnist,cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, merge, Lambda,Highway,GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Reshape
from keras.utils import np_utils
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.optimizers import SGD
from keras.layers.noise import GaussianNoise
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense
from keras.layers import Reshape, Lambda, concatenate, GlobalAveragePooling1D, Add
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


from data_loader import DataGenerator
from keras.callbacks import ModelCheckpoint
from schedules import onetenth_50_75



from data_loader import DataGenerator
from model import MatMul, mySign, myConv, myAgg






input_points = Input(shape=(2048, 3))
# issues
# input transformation net
x = Conv1D(64, 1, activation='relu')(input_points)
x = BatchNormalization()(x)
x = Conv1D(128, 1, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(1024, 1, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2048)(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)

x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
input_T = Reshape((3, 3))(x)

# forward net0
g_0 = MatMul()([input_points, input_T])
g_0_offset = Conv1D(64, 1, activation='relu', 
                    kernel_initializer='zeros', bias_initializer='zeros')(g_0)
g_0_mul = Conv1D(64, 1, activation='relu')(g_0)
g_0_mul = BatchNormalization()(g_0_mul)
g_1 = Add()([g_0_mul, g_0_offset])



g_1_offset = Conv1D(128, 1, activation='relu',
                    kernel_initializer='zeros', bias_initializer='zeros')(g_1)
g_1_mul = Conv1D(128, 1, activation='relu')(g_1)
g_1_mul = BatchNormalization()(g_1_mul)
g_2 = Add()([g_1_mul, g_1_offset])


g_2_offset = Conv1D(256, 1, activation='relu',
                    kernel_initializer='zeros', bias_initializer='zeros')(g_2)
g_2_mul = Conv1D(256, 1, activation='relu')(g_2)
g_2_mul = BatchNormalization()(g_2_mul)
g_3 = Add()([g_2_mul, g_2_offset])


g_3_offset = Conv1D(512, 1, activation='relu',
                    kernel_initializer='zeros', bias_initializer='zeros')(g_3)
g_3_mul = Conv1D(512, 1, activation='relu')(g_3)
g_3_mul = BatchNormalization()(g_3_mul)
g_4 = Add()([g_3_mul, g_3_offset])


g_4_offset = Conv1D(64, 1, activation='relu',
                    kernel_initializer='zeros', bias_initializer='zeros')(g_4)
g_4_mul = Conv1D(64, 1, activation='relu')(g_4)
g_4_mul = BatchNormalization()(g_4_mul)
g_5 = Add()([g_4_mul, g_4_offset])


# global feature
global_feature = MaxPooling1D(pool_size=2048)(g_5)

# point_net_cls
c = Dense(512, activation='relu')(global_feature)
c = BatchNormalization()(c)
c = Dropout(0.5)(c)
c = Dense(256, activation='relu')(c)
c = BatchNormalization()(c)
c = Dropout(0.5)(c)
c = Dense(40, activation='softmax')(c)
prediction = Flatten()(c)

model = Model(inputs=input_points, outputs=[prediction])


xx = np.random.rand(32,2048, 3) - 0.5
y = model.predict_on_batch(xx)



nb_classes = 40
train_file = '/home/changetest/datasets/Modelnet40/ply_data_train.h5'
test_file = '/home/changetest/datasets/Modelnet40/ply_data_test.h5'

epochs = 100
batch_size = 32

train = DataGenerator(train_file, batch_size, nb_classes, train=True)
val = DataGenerator(test_file, batch_size, nb_classes, train=False)

model.summary()
lr = 0.0001
adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


if not os.path.exists('./results/'):
    os.mkdir('./results/')
checkpoint = ModelCheckpoint('./results/pointnet.h5', monitor='val_acc',
                             save_weights_only=True, save_best_only=True,verbose=1)
model.fit_generator(train.generator(), steps_per_epoch=9840 // batch_size, 
                    epochs=epochs, validation_data=val.generator(), 
                    callbacks=[checkpoint, onetenth_50_75(lr)],
                    validation_steps=2468 // batch_size, verbose=1)
































