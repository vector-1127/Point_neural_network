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

class GumbelSoftmax(Layer):
    def __init__(self, temperature, hard = False, **kwargs):
        self.temperature = temperature
        self.hard = hard
        super(GumbelSoftmax, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, inputs):
        _, _, self.nb_feature  = tf.unstack(tf.shape(inputs[0]), num=3)
        self.nb_batch, self.nb_dim, self.nb_cluster  = tf.unstack(tf.shape(inputs[1]), num=3)
        values = inputs[0]
        indices = self._get_indices(inputs[1])
        #return indices
        return self._batchwise_unsorted_segment(values, indices)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][2], input_shape[0][2])
    
    def _get_indices(self, x):
        nb_batch, nb_dim, nb_cluster  = tf.unstack(tf.shape(x), num=3)
        y = self.gumbel_softmax_sample(x, self.temperature)
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        ####
        clusters = tf.cast(tf.range(nb_cluster), tf.float32)
        out = tf.reduce_sum(y*clusters, axis = 2, keep_dims = True)
        return tf.cast(out, tf.int32)
     
    
    def _batchwise_unsorted_segment(self, values, indices):
        """UnsortedSegmentSum on each row.
        Args:
          values: a `Tensor` with shape `[nb_batch, nb_dim, nb_feature]`.
          indices: an integer `Tensor` with shape `[nb_batch, nb_dim]`.
          n: an integer.
        Returns:
          A `Tensor` with the same type as `values` and shape `[nb_batch, nb_dim, nb_cluster]`.
        """
        indices = tf.tile(indices,[1,1,self.nb_feature]) + tf.range(self.nb_feature) * self.nb_cluster
        values = tf.reshape(values,[self.nb_batch,-1])
        indices = tf.reshape(indices,[self.nb_batch,-1])
        res = tf.reshape(self._rowwise_unsorted_segment(values, indices, self.nb_feature*self.nb_cluster),
                         [self.nb_batch, self.nb_feature, self.nb_cluster])
        return tf.transpose(res, perm=[0,2,1])
    
    
    def _rowwise_unsorted_segment(self, values, indices, n):
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
    
    
    
    
    
    
    