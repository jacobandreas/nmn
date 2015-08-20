#!/usr/bin/env python2

from util import Index

from data.util import pp
import logging
from lasagne import init, layers, objectives, updates, nonlinearities
import numpy as np
import theano
import theano.tensor as T
import warnings

class TileLayer(layers.Layer):
  def __init__(self, incoming, reps, **kwargs):
    super(TileLayer, self).__init__(incoming, **kwargs)
    self.reps = reps

  def get_output_shape_for(self, input_shape):
    output_shape = []
    for i in range(max(len(input_shape), len(self.reps))):
      rep = self.reps[i] if i < len(self.reps) else 1
      inp = input_shape[i] if i < len(input_shape) else 1
      output_shape.append(rep * inp)
    return tuple(output_shape)

  def get_output_for(self, input, **kwargs):
    return T.tile(input, self.reps)

class LSTM:
  def __init__(self, input_type, output_type, params):
    batch_size = params["batch_size"]
    #world_size = params["world_size"]
    image_size = params["image_size"]
    channels = params["channels"]
    vocab_size = params["vocab_size"]
    hidden_size = params["hidden_size"]
    output_size = params["output_size"]

    self.batch_size = batch_size
    self.query_max_length = params["query_max_length"]

    self.l_input_text = layers.InputLayer((batch_size, self.query_max_length))
    self.l_input_world = layers.InputLayer((batch_size, channels, image_size,
        image_size))

    self.l_embed_text = layers.EmbeddingLayer(self.l_input_text, vocab_size, hidden_size)

    #self.l_dense_world_1 = layers.DenseLayer(self.l_input_world, hidden_size)
    #self.l_dense_world_2 = layers.DenseLayer(self.l_dense_world_1, hidden_size)
    self.l_conv = layers.Conv2DLayer(self.l_input_world, channels, 5, border_mode="same")

    #self.l_reshape_world = layers.ReshapeLayer(self.l_dense_world_2, (batch_size, 1, hidden_size))
    #self.l_concat = layers.ConcatLayer([self.l_reshape_world, self.l_embed_text])

    #self.l_reshape_world = layers.ReshapeLayer(self.l_dense_world_2, (batch_size, 1, hidden_size))

    self.l_reshape_world = layers.ReshapeLayer(self.l_conv, (batch_size, 1, -1))
    self.l_tile_world = TileLayer(self.l_reshape_world, (1, self.query_max_length, 1))
    self.l_concat = layers.ConcatLayer([self.l_embed_text, self.l_tile_world], axis=2)

    self.l_forward_1 = layers.LSTMLayer(self.l_concat, hidden_size)
    self.l_forward_2 = layers.LSTMLayer(self.l_forward_1, hidden_size)
    self.l_slice = layers.SliceLayer(self.l_forward_2, indices=-1, axis=1)

    #self.l_predict = layers.DenseLayer(self.l_slice, output_size, nonlinearity=None)
    self.l_predict = layers.DenseLayer(self.l_slice, output_size, nonlinearity=nonlinearities.softmax)

    self.t_input_text = T.imatrix("input_text")
    self.t_input_world = input_type.make_input() #T.matrix("input_world")
    self.input_mapping = {self.l_input_text: self.t_input_text,
                          self.l_input_world: self.t_input_world}
    self.t_output = layers.get_output(self.l_predict, self.input_mapping)
    self.t_target = output_type.make_target() #T.matrix("target")
    self.t_loss = objectives.aggregate(output_type.loss(self.t_output, self.t_target))
    self.t_pred = T.argmax(self.t_output, axis=1)
    self.params = layers.get_all_params(self.l_predict)

    grads = T.grad(self.t_loss, self.params)
    scaled_grads = updates.total_norm_constraint(grads, 1.)

    momentum = 0.9
    lr = .1
    upd = updates.adadelta(scaled_grads, self.params, learning_rate = lr)

    loss_inputs = [self.t_input_text, self.t_input_world, self.t_target]
    self.f_loss = theano.function(loss_inputs, self.t_loss)
    self.f_train = theano.function(loss_inputs, self.t_loss, updates=upd)
    self.f_pred = theano.function([self.t_input_text, self.t_input_world], self.t_pred)

    print "done"

    self.index = Index()
    self.index.index("NULL")

  def linearize(self, query):
    pretty = pp(query)
    sep = pretty.replace("(", "( ").replace(")", " )")
    out = np.zeros((self.batch_size, self.query_max_length), dtype=np.int32)
    for i, t in enumerate(sep.split()):
      out[:,i] = self.index.index(t)

    return out
    #indices = [self.index.index(t) for t in sep.split()]
    #return np.tile(indices, (self.batch_size, 1)).astype(np.int32)

  def train(self, query, world, output, return_norm = False):
    lq = self.linearize(query)
    return self.f_train(lq, world, output)

  def loss(self, query, world, output):
    lq = self.linearize(query)
    return self.f_loss(lq, world, output)

  def predict(self, query, world):
    lq = self.linearize(query)
    return self.f_pred(lq, world)

  def serialize(self, dest):
    pass
