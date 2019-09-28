#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:02:11 2018

@author: changetest
"""


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from schedules import onetenth_35_70



from data_loader import DataGenerator
from model import GumbelSoftmax, MatMul, GumbelPooling, GumbelIntegration

from keras import regularizers




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

# forward net
g = MatMul()([input_points, input_T])
g = Conv1D(64, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = Conv1D(64, 1, activation='relu')(g)
g = BatchNormalization()(g)


# feature transform net
f = Conv1D(64, 1, activation='relu')(g)
f = BatchNormalization()(f)
f = Conv1D(128, 1, activation='relu')(f)
f = BatchNormalization()(f)
f = Conv1D(1024, 1, activation='relu')(f)
f = BatchNormalization()(f)
f = MaxPooling1D(pool_size=2048)(f)
f = Dense(512, activation='relu')(f)
f = BatchNormalization()(f)
f = Dense(256, activation='relu')(f)
f = BatchNormalization()(f)
f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
feature_T = Reshape((64, 64))(f)

in_0 = MatMul()([g, feature_T])

# forward net0
f_0 = Conv1D(128, 1, activation='relu')(in_0)
f_0 = BatchNormalization()(f_0)
c_0 = Conv1D(64, 1, activation='relu')(in_0)
c_0 = BatchNormalization()(c_0)
c_0 = GumbelSoftmax(temperature=0.1, hard = False)(c_0)
i_0 = GumbelSoftmax(temperature=1, hard = True)(c_0)
for _ in range(8):
    i_0_t = GumbelSoftmax(temperature=1, hard = True)(c_0)
    i_0 = GumbelIntegration('max')([i_0,i_0_t])
out_0 = GumbelPooling(pool_way = 'mean')([f_0, i_0])


# forward net1
f_1 = Conv1D(256, 1, activation='relu')(out_0)
f_1 = BatchNormalization()(f_1)
c_1 = Conv1D(16, 1, activation='relu')(out_0)
c_1 = BatchNormalization()(c_1)
c_1 = GumbelSoftmax(temperature=0.1, hard = False)(c_1)
i_1 = GumbelSoftmax(temperature=1, hard = True)(c_1)
for _ in range(8):
    i_1_t = GumbelSoftmax(temperature=1, hard = True)(c_1)
    i_1 = GumbelIntegration('max')([i_1,i_1_t])
out_1 = GumbelPooling(pool_way = 'mean')([f_1, i_1])



# forward net1
f_2 = Conv1D(512, 1, activation='relu')(out_1)
f_2 = BatchNormalization()(f_2)
c_2 = Conv1D(4, 1, activation='relu')(out_1)
c_2 = BatchNormalization()(c_2)
c_2 = GumbelSoftmax(temperature=0.1, hard = False)(c_2)
i_2 = GumbelSoftmax(temperature=1, hard = True)(c_2)
for _ in range(8):
    i_2_t = GumbelSoftmax(temperature=1, hard = True)(c_2)
    i_2 = GumbelIntegration('max')([i_2,i_2_t])
out_2 = GumbelPooling(pool_way = 'mean')([f_2, i_2])


''''''
global_feature = MaxPooling1D(pool_size=4)(out_2)
c = Dense(512, activation='relu')(global_feature)
c = BatchNormalization()(c)
c = Dropout(0.5)(c)
c = Dense(256, activation='relu')(c)
c = BatchNormalization()(c)
c = Dropout(0.5)(c)
c = Dense(40, activation='softmax')(c)
prediction = Flatten()(c)

'''
model = Model(inputs=input_points, outputs=[prediction])
xx = np.random.rand(32,2048, 3) - 0.5
y = model.predict_on_batch(xx)
'''

model = Model(inputs=input_points, outputs=[prediction])
nb_classes = 40
train_file = '/home/changetest/datasets/Modelnet40/ply_data_train.h5'
test_file = '/home/changetest/datasets/Modelnet40/ply_data_test.h5'

epochs = 100
batch_size = 32

train = DataGenerator(train_file, batch_size, nb_classes, train=True)
val = DataGenerator(test_file, batch_size, nb_classes, train=False)

#model.summary()
lr = 0.0001
adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


if not os.path.exists('./results/'):
    os.mkdir('./results/')
checkpoint = ModelCheckpoint('./results/pointnet.h5', monitor='val_acc',
                             save_weights_only=True, save_best_only=True,verbose=1)
model.fit_generator(train.generator(), steps_per_epoch=9840 // batch_size, 
                    epochs=epochs, validation_data=val.generator(), 
                    callbacks=[checkpoint, onetenth_35_70(lr)],
                    validation_steps=2468 // batch_size, verbose=2)







'''
f = Model(inputs=input_points, outputs=[c_0, i_0])
f_out = f.predict_generator(val.generator(), 50)
'''




















