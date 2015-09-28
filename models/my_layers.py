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

class Accuracy(Layer):
    def __init__(self, name, **kwargs):
        super(Accuracy, self).__init__(self, name, kwargs)
        self.kwargs = kwargs
        self.p.type = "Py"

    def setup(self, bttom, top):
        top[0].reshape((1,))

    def forward(self, bottom, top):
        output, target = bottom
        prediction = np.argmax(output.data, axis=1)
        agreements = np.sum(prediction == target.data)
        top[0].data[...] = np.asarray(agreements)
        return agreements

    def backward(self, top, bottom):
        pass
        #raise NotImplementedError()

class Collapse(Layer):
    def __init__(self, name, **kwargs):
        super(Collapse, self).__init__(self, name, kwargs)
        self.kwargs = kwargs
        self.p.type = "Py"

    def setup(self, bottom, top):
        input = T.tensor4("input")
        v = T.matrix("v")
        result = T.sum(input, axis=(2,3))
        result_g = T.Lop(result, input, v)
        self.f = theano.function([input], result)
        self.b = theano.function([input, v], result_g)

    def forward(self, bottom, top):
        input = bottom[0]
        top[0].reshape(input.shape[0:2])
        top[0].data[...] = self.f(input.data)
        return 0

    def backward(self, top, bottom):
        input = bottom[0]
        v = top[0].diff
        input.diff[...] += self.b(input.data, v)

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

class Index(Layer):
    def __init__(self, name, index, **kwargs):
        super(Index, self).__init__(self, name, kwargs)
        self.kwargs = kwargs
        self.p.type = "Py"

        self.index = index

    def setup(self, bottom, top):
        pass

    def forward(self, bottom, top):
        vector_size = bottom[0].shape[1]
        top[0].reshape((1, vector_size,))
        top[0].data_tensor.copy_chunk_from(
                bottom[0].data_tensor, vector_size, 0, vector_size * self.index)
        return 0

    def backward(self, top, bottom):
        vector_size = top[0].shape[1]
        bottom[0].diff_tensor.copy_chunk_from(
                top[0].diff_tensor, vector_size, vector_size * self.index, 0)


class BroadcastSum(Layer):
    def __init__(self, name, **kwargs):
        super(BroadcastSum, self).__init__(self, name, kwargs)
        self.kwargs = kwargs
        self.p.type = "Py"

    def setup(self, bottom, top):
        small_size = bottom[0].shape[1]
        small = T.matrix("small")
        big = T.tensor4("big")
        v = T.tensor4("v")
        small_bc = small.dimshuffle(0, 1, "x", "x")
        small_bc = T.addbroadcast(small_bc, 0)
        result = big + small_bc

        g_small, g_big = T.Lop(result, [small, big], v)
        self.f = theano.function([small, big], result)
        self.b_small = theano.function([v], g_small)
        self.b_big = theano.function([v], g_big)

    def forward(self, bottom, top):
        small, big = bottom
        top[0].reshape(big.shape)
        top[0].data[...] = self.f(small.data, big.data)
        return 0

    def backward(self, top, bottom):
        small, big = bottom
        v = top[0].diff
        small.diff[...] += self.b_small(v)
        big.diff[...] += self.b_big(v)

class FeatureDot(Layer):
    def __init__(self, name, **kwargs):
        super(FeatureDot, self).__init__(self, name, kwargs)
        self.kwargs = kwargs
        self.p.type = "Py"

    def setup(self, bottom, top):
        a = T.tensor4("a")
        b = T.tensor4("b")
        v = T.tensor3("v")

        dot = a * b
        result = T.sum(dot, axis=1)

        g_a, g_b = T.Lop(result, [a, b], v)
        self.f = theano.function([a, b], result)
        self.b_a = theano.function([a, b, v], g_a)
        self.b_b = theano.function([a, b, v], g_b)

    def forward(self, bottom, top):
        a, b = bottom
        result = self.f(a.data, b.data)
        nb, nr, nc = result.shape
        result.shape = (nb, 1, nr, nc)
        top[0].reshape(result.shape)
        top[0].data[...] = result
        return 0

    def backward(self, top, bottom):
        nb, nf, nr, nc = bottom[0].shape
        # TODO why is this necessary?
        top[0].reshape((nb, nr, nc))
        a, b = bottom
        v = top[0].diff
        a.diff[...] += self.b_a(a.data, b.data, v)
        b.diff[...] += self.b_b(a.data, b.data, v)
