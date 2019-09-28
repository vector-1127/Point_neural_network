from keras.layers.core import *
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, regularizers, constraints


class ScatterNet(Layer):
    '''
    Input shape: (batch_size, nb_node, 1, nb_feature)
    Output shape: (batch_size, nb_node, nb_scatter, nb_feature)
    '''
    def __init__(self, nb_scatter, scattering_function = 'chebyshev', **kwargs):
        if K.backend() != 'tensorflow':
            raise Exception("GraphConv Requires Tensorflow Backend.")
        self.nb_scatter = nb_scatter
        self.scattering_function = getattr(self, scattering_function)
        super(ScatterNet, self).__init__(**kwargs)
        
                
    def build(self, input_shape):
        self.nb_feature = input_shape[3]
        self.nb_node = input_shape[1]
        self.built = True
        
        
    def call(self, x, mask = None):
        # (batch_size, nb_node, 1, nb_feature) => (batch_size, nb_node, nb_scatter, nb_feature)
        return self.scattering_function(x)
        
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_node, self.nb_scatter, self.nb_feature)
        
    
        
    def chebyshev(self, x):
        # Chebyshev polynomial: [-1,1]
        if self.nb_scatter > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = 2 * x # x or 2 * x
            y = tf.concat(axis = 2, values = [x0,x1])
        for _ in range(2, self.nb_scatter):
            x2 = 2 * x * x1 - x0
            y = tf.concat(axis = 2, values = [y,x2])
            x0, x1 = x1, x2
        return y
    
    def legendre(self,x):
        # Legendre Polynomials : [-1,1]
        x = tf.expand_dims(x, dim = 3)
        if self.nb_scatter > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = x
            y = tf.concat(axis = 2, values = [x0,x1])
        for n in range(2, self.nb_scatter):
            x2 = 1.0*(2*n-1)/(n) * x * x1 - 1.0*(n-1)/(n)*x0
            y = tf.concat(axis = 2, values = [y,x2])
            x0, x1 = x1, x2
        return y
        
    def laguerre(self,x):
        # Laguerre polynomial: [0,+infinity]
        x = tf.expand_dims(x, dim = 3)
        if self.nb_scatter > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = x0 - x
            y = tf.concat(axis = 2, values = [x0,x1])
        for n in range(2, self.nb_scatter):
            x2 = (2 * n -1) * x1 - x * x1 - (n-1) * (n-1) * x0
            y = tf.concat(axis = 2, values = [y,x2])
            x0, x1 = x1, x2
        return y
        
    def hermite(self, x):
        # Hermite polynomial: [-infinity,+infinity]
        x = tf.expand_dims(x, dim = 3)
        if self.nb_scatter > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = 2 * x
            y = tf.concat(axis = 2, values = [x0,x1])
        for n in range(2, self.nb_scatter):
            x2 = 2 * x * x1 - 2 * (n-1) * x0
            y = tf.concat(axis = 2, values = [y,x2])
            x0, x1 = x1, x2
        return y
        
    def optimum(self, x):
        # Optimum polynomial: {1, x, x^{2}, x^{3}, ...,}
        x = tf.expand_dims(x, dim = 3)
        if self.nb_scatter > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = x
            y = tf.concat(axis = 2, values = [x0,x1])
        for n in range(2, self.nb_scatter):
            x2 = x * x1
            y = tf.concat(axis = 2, values = [y,x2])
            x0, x1 = x1, x2
        return y
















