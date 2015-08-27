#!/usr/bin/env python2

from apollocaffe.layers.layer_headers import Layer, PyLayer
import numpy as np

import theano
import theano.tensor as T

class Tile(Layer):
    def __init__(self, name, **kwargs):
        super(Tile, self).__init__(self, name, kwargs)

class Flatten(Layer):
    def __init__(self, name, **kwargs):
        super(Flatten, self).__init__(self, name, kwargs)

class Reshape(Layer):
    def __init__(self, name, **kwargs):
        super(Reshape, self).__init__(self, name, kwargs)


class Attention(Layer):
    def __init__(self, name, **kwargs):
        super(Attention, self).__init__(self, name, kwargs)
        self.kwargs = kwargs
        self.p.type = "Py"
        
    def setup(self, bottom, top):
        attention = T.tensor4("attention")
        input = T.tensor4("input")
        v = T.matrix("v")
        attention_bc = T.addbroadcast(attention, 1)
        attended = T.mul(input, attention_bc)
        result = T.sum(attended, axis=(2,3))
        result_g_attention, result_g_input = T.Lop(result, [attention, input], v)
        self.f = theano.function([attention, input], result)
        self.b_attention = theano.function([attention, input, v], result_g_attention)
        self.b_input = theano.function([attention, input, v], result_g_attention)

    def forward(self, bottom, top):
        attention, input = bottom
        top[0].reshape(input.shape[0:2])
        top[0].data[...] = self.f(attention.data, input.data)
        return 0

    def backward(self, top, bottom):
        attention, input = bottom
        v = top[0].diff
        attention.diff[...] += self.b_attention(attention.data, input.data, v)
        input.diff[...] += self.b_input(attention.data, input.data, v)
