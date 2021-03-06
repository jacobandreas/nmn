#!/usr/bin/env python2

from apollocaffe.layers.layer_headers import Layer, PyLayer
from apollocaffe import Tensor
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
        self.pythonargs = []
        self.p.type = "Py"

    def setup(self, bttom, top):
        top[0].reshape((1,))

    def forward(self, bottom, top):
        output, target = bottom
        prediction = np.argmax(output.data, axis=1)
        agreements = np.mean(prediction == target.data)
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
        self.pythonargs = []
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

#class FeatureDot(Layer):
#    def __init__(self, name, **kwargs):
#        super(FeatureDot, self).__init__(self, name, kwargs)
#        self.kwargs = kwargs
#        self.p.type = "Py"
#
#    def setup(self, bottom, top):
#        pass
#
#    def forward(self, bottom, top):
#        weights, features = bottom
#        n_batch, n_channels, n_rows, n_cols = features.data_tensor.shape
#        tmp = Tensor()
#        tmp.reshape((n_batch, n_channels))
#        top[0].reshape((n_batch, 1, n_rows, n_cols))
#        for i_row in range(n_rows):
#            for i_col in range(n_cols):
#                tmp.CopyChunkFrom(
#                top[0].data_tensor[:,0,i_row,i_col] = weights.data_tensor * features.data_tensor[:,:,i_row,i_col]
#
#        exit()
#
#    def backward(self, top, bottom):
#        pass

class FeatureDot(Layer):
    def __init__(self, name, **kwargs):
        super(FeatureDot, self).__init__(self, name, kwargs)
        self.kwargs = kwargs
        self.p.type = "Py"

    def setup(self, bottom, top):
        weights = T.matrix("weights")
        weights_bc = weights.dimshuffle((0, 1, "x", "x"))
        feats = T.tensor4("weights")
        v = T.tensor3("v")

        dot = weights_bc * feats
        result = T.sum(dot, axis=1)

        g_w, g_f = T.Lop(result, [weights, feats], v)
        self.f = theano.function([weights, feats], result)
        self.b_w = theano.function([weights, feats, v], g_w)
        self.b_f = theano.function([weights, feats, v], g_f)

    def forward(self, bottom, top):
        weights, feats = bottom
        result = self.f(weights.data, feats.data)
        nb, nr, nc = result.shape
        result.shape = (nb, 1, nr, nc)
        top[0].reshape(result.shape)
        top[0].data[...] = result
        return 0

    def backward(self, top, bottom):
        weights, feats = bottom
        nb, nf, nr, nc = feats.shape
        # TODO why is this necessary?
        top[0].reshape((nb, nr, nc))
        v = top[0].diff
        weights.diff[...] += self.b_w(weights.data, feats.data, v)
        feats.diff[...] += self.b_f(weights.data, feats.data, v)

#class FeatureDot(Layer):
#    def __init__(self, name, **kwargs):
#        super(FeatureDot, self).__init__(self, name, kwargs)
#        self.kwargs = kwargs
#        self.p.type = "Py"
#
#    def setup(self, bottom, top):
#        pass
#
#    def forward(self, bottom, top):
#        weights, feats = bottom
#        n_b, n_f, n_r, n_c = feats.shape
#        work = Tensor()
#        work.reshape((n_b, n_f))
#        for r in range(n_r):
#            for c in range(n_c):
#                work.copy_chunk_from(feats, n_b * n_f, 
#                print dir(work)
#                exit()
#        return 0
#
#    def backward(self, top, bottom):
#        pass
