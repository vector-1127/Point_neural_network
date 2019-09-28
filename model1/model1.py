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
        
    def call(self, inputs):
        nb_batch, nb_dim, nb_clu  = tf.unstack(tf.shape(inputs), num=3)
        y = gumbel_softmax_sample(inputs, self.temperature)
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        ####
        clusters = tf.constant(np.asarray(np.arange(self.nb_cluster)), tf.float32)
        out = tf.reduce_sum(y*clusters, axis = 2, keep_dims = True)
        return tf.cast(out, tf.int32)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)

        
        
class GumbelPooling(Layer):
    def __init__(self, **kwargs):
        super(GumbelPooling, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.built = True
    
    def call(self, inputs):
        values, indices = inputs[0], inputs[1]
        return y



        
        
def _rowwise_unsorted_segment_sum(values, indices, n):
    """UnsortedSegmentSum on each row.
    Args:
      values: a `Tensor` with shape `[batch_size, k]`.
      indices: an integer `Tensor` with shape `[batch_size, k]`.
      n: an integer.
    Returns:
      A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
    """
    batch, k = tf.unstack(tf.shape(indices), num=2)
    indices_flat = tf.reshape(indices, [-1]) + tf.div(tf.range(batch * k), k) * n
    ret_flat = tf.unsorted_segment_sum(
        tf.reshape(values, [-1]), indices_flat, batch * n)
    return tf.reshape(ret_flat, [batch, n])

        
def _batchwise_unsorted_segment(values, indices, nb_cluster):
    """UnsortedSegmentSum on each row.
    Args:
      values: a `Tensor` with shape `[nb_batch, nb_dim, nb_feature]`.
      indices: an integer `Tensor` with shape `[nb_batch, nb_dim]`.
      n: an integer.
    Returns:
      A `Tensor` with the same type as `values` and shape `[nb_batch, nb_dim, nb_cluster]`.
    """
    nb_batch, nb_dim, nb_feature  = tf.unstack(tf.shape(values), num=3)
    indices = tf.tile(indices,[1,1,nb_feature]) + tf.range(nb_feature) * nb_cluster
    values = tf.reshape(values,[nb_batch,-1])
    indices = tf.reshape(indices,[nb_batch,-1])
    res = tf.reshape(_rowwise_unsorted_segment_sum(values, indices, nb_feature*nb_cluster),[nb_batch, nb_feature, nb_cluster])
    return tf.transpose(res, perm=[0,2,1])



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
    
    
    x = tf.placeholder(tf.int32, [None, 4, 2])
    i = tf.placeholder(tf.int32, [None, 4, 1])
    f = K.function([x,i],[_batchwise_unsorted_segment(x,i,3)])
    
    xx = np.random.randint(0,2,[3,4,2])
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
    
    
    
    
    
    
    