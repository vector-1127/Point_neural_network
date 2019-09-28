#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:47:33 2018

@author: changetest
"""

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense
from keras.layers import Reshape, Lambda, concatenate
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras import backend as K

from distributions import gumbel_softmax_sample

class GumbelSoftmax(Layer):
    def __init__(self, temperature, hard = False, **kwargs):
        self.temperature = temperature
        self.hard = hard
        super(GumbelSoftmax, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.nb_cluster = input_shape[2]
        self.built = True
        
    def call(self, x):
        y = gumbel_softmax_sample(x, self.temperature)
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        ####
        clusters = tf.constant(np.asarray(np.arange(self.nb_cluster)), tf.float32)
        out = tf.reduce_sum(y*clusters,axis = 2)
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])



'''
def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
'''



class myConv(Layer):
    def __init__(self, **kwargs):
        super(myConv, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, x):
        input_, filter_ = x[0], x[1]
        return input_*filter_
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]




class myAgg(Layer):
    def __init__(self, **kwargs):
        super(myAgg, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, x):
        input_, location_ = x[0], x[1]
        exp = np.array([2**i for i in range(location_.get_shape()[2])])
        y, idx = tf.unique(K.sum(location_*exp, axis = 1))
       # self.node_ = tf.shape(y)[0]
        #res = tf.unsorted_segment_max(input_, idx, self.node_)
        return y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])





class mySign(Layer):
    def __init__(self, threshold=0., **kwargs):
        self.threshold = K.cast_to_floatx(threshold)
        super(mySign, self).__init__(**kwargs)

    def call(self, x):
        return K.sign((K.sign(x - self.threshold)+1)/2)

    def compute_output_shape(self, input_shape):
        return input_shape


class myNeighbor(Layer):
    def __init__(self, nb_neighbor, **kwargs):
        self.nb_neighbor = nb_neighbor
        super(myNeighbor, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, x):
        input_x, center, size = x[0], x[1], x[2]
        # dist = tf.abs(x - center)
        #yorn = 1 - tf.sign(K.sum(tf.sign(dist-size),axis = 2) + 3)
        return center
        #return x * tf.expand_dims(yorn, axis = 2)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.nb_neighbor)

class updateCenter(Layer):
    def __init__(self, **kwargs):
        super(updateCenter, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, x):
        center, offset = x[0], x[1]
        return center + offset
    
    def compute_output_shape(self, input_shape):
        return input_shape



class MatMul(Layer):

    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`MatMul` layer should be called '
                             'on a list of inputs')
        if len(input_shape) != 2:
            raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('The dimensions of each element of inputs should be 3')

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatMul` layer should be called '
                             'on a list of inputs.')
        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)


def PointNet(nb_classes):
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

    # forward net
    g = MatMul()([g, feature_T])
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global feature
    global_feature = MaxPooling1D(pool_size=2048)(g)

    # point_net_cls
    c = Dense(512, activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(nb_classes, activation='softmax')(c)
    prediction = Flatten()(c)

    model = Model(inputs=input_points, outputs=prediction)

    return model

if __name__ == "__main__":
    
    
    x = tf.placeholder(tf.float32, [None, 10000, 3])
    sorting_w = tf.placeholder(tf.float32, [None, 10000, 10000])
    
    '''
    data_id = np.asarray([2*i for i in range(5)])
    data_id.shape = -1,1,1
    mask1 = data_id + mask
    
    res = tf.boolean_mask(x, mask)
    
    
    res = tf.unsorted_segment_max(x, mask1, 5*2)
    '''
    
    
    y = tf.matmul(sorting_w, x)
    
    f = K.function(inputs = [x, sorting_w], outputs =[y])
    
    x_ = np.random.rand(32,10000, 3)
    w_ = np.random.rand(32,10000, 10000)
    
    for i in range(10000):
        print(i)
        y = f([x_, w_])
    
    
    
    
    
    
    
    