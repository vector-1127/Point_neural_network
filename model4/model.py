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

#from distributions import gumbel_softmax_sample


class GumbelIntegration(Layer):
    '''
    input[0]: feature: B*N*C
    input[1]: cluster: B*N*C
    '''
    def __init__(self, inte_way = 'max', **kwargs):
        self.inte_way = getattr(self, inte_way+'_')
        super(GumbelIntegration, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, inputs):
        x0 = tf.expand_dims(inputs[0], axis = 0)
        x1 = tf.expand_dims(inputs[1], axis = 0)
        return self.inte_way(tf.concat([x0, x1], axis = 0))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])
    
    def max_(self, x):
        return tf.reduce_max(x, axis = 0)



class GumbelPooling(Layer):
    '''
    input[0]: feature: B*N*F 
    input[1]: cluster: B*N*C
    '''
    def __init__(self, pool_way = 'max', **kwargs):
        self.pool_way = getattr(self, pool_way+'_')
        super(GumbelPooling, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, inputs):
        x0 = tf.expand_dims(inputs[0], axis = -1)
        x1 = tf.expand_dims(inputs[1], axis = 2)
        y = tf.transpose(self.pool_way(x0*x1), perm =[0, 2, 1])
        return y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][2], input_shape[0][2])
    
    def max_(self, x):
        return tf.reduce_max(x, axis = 1)





class GumbelSoftmax(Layer):
    def __init__(self, temperature, hard = False, **kwargs):
        self.temperature = temperature
        self.hard = hard
        super(GumbelSoftmax, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.clusters = input_shape[2]
        self.built = True
        
    def call(self, inputs):
        return self._get_indices(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def _get_indices(self, x):
        y = self.gumbel_softmax_sample(x, self.temperature)
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y
     
    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        return -tf.log(-tf.log(U + eps) + eps)
    
    
    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature)
    
    
    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_dim, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_dim,  n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probability distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y
    


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
  
        
        
        
        
if __name__ == "__main__":
    
    temp = GumbelSoftmax(1)
    
    x = tf.placeholder(tf.float32, [None, 4, 2])
    i = tf.placeholder(tf.int32, [None, 4, 1])
    f = K.function([x,i],[temp._batchwise_unsorted_segment(x,i,3)])
    
    xx = np.random.rand(3,4,2)
    ii = np.random.randint(0,3,[3,4,1])
    y = f([xx,ii])
    
    
    
    '''
    sorting_w = tf.placeholder(tf.float32, [None, 10000, 10000])
    
    
    data_id = np.asarray([2*i for i in range(5)])
    data_id.shape = -1,1,1
    mask1 = data_id + mask
    
    res = tf.boolean_mask(x, mask)
    
    
    res = tf.unsorted_segment_max(x, mask1, 5*2)
    
    
    
    y = tf.matmul(sorting_w, x)
    
    f = K.function(inputs = [x, sorting_w], outputs =[y])
    
    x_ = np.random.rand(32,10000, 3)
    w_ = np.random.rand(32,10000, 10000)
    
    for i in range(10000):
        print(i)
        y = f([x_, w_])
    '''
    
    
    
    
    
    
    