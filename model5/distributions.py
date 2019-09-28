#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:49:30 2018

@author: changetest
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K

from collections import namedtuple

try:
    from tensorflow.contrib.distributions.python.ops.relaxed_onehot_categorical import ExpRelaxedOneHotCategorical
except:
    from tensorflow.contrib.distributions.python.ops.relaxed_onehot_categorical import _ExpRelaxedOneHotCategorical
    print("TensorFlow native concrete distribution (this version doesn't work).")


def sample_gumbel(shape, eps=1e-10):
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
        logits: [batch_size, n_dim, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_dim,  n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(tf.log(tf.nn.softmax(logits, dim = 2)), temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y



if __name__ == "__main__":
    
    x = tf.placeholder(tf.float32,shape=[None,3,2])
    y = gumbel_softmax(x, 0.01, hard=True)
    f = K.function([x],[y])
    
    xx = np.random.rand(4,3,2)
    print(xx)
    
    yy = 0
    for _ in range(1000):
        yy += f([xx])[0]
    print(yy/1000)
    
    
    
    
    
    