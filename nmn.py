#!/usr/bin/env python2

from data.util import pp

import logging
from lasagne import init, layers, objectives, updates, nonlinearities
import numpy as np
import theano
import theano.tensor as T
import warnings

class Network:
  def __init__(self, l_inputs, l_output):
    self.l_inputs = l_inputs
    self.l_output = l_output

  def add_objective(self, input_type, output_type):
    params = layers.get_all_params(self.l_output)
    self.params = params

    t_input = input_type.make_input()
    input_mapping = {l_input: t_input for l_input in self.l_inputs}
    t_output = layers.get_output(self.l_output, input_mapping)
    t_pred = T.argmax(t_output, axis=1)
    t_target = output_type.make_target()

    momentum = 0.9
    lr = .001

    t_loss = objectives.aggregate(output_type.loss(t_output, t_target))
    upd = updates.adam(t_loss, params)
    loss_inputs = [t_input, t_target]
    self.loss = theano.function(loss_inputs, t_loss)
    self.train = theano.function(loss_inputs, t_loss, updates=upd)
    self.predict = theano.function([t_input], t_pred)

def wire(query, modules):
  if not isinstance(query, tuple):
    return modules.get(query).instantiate()
  args = [wire(q, modules) for q in query[1:]]
  return modules.get(query[0]).instantiate(*args)

class IdentityModule:
  def __init__(self):
    pass

  def instantiate(self, *inputs):
    assert len(inputs) == 1
    return inputs[0]


class ConvModule:
  def __init__(self, batch_size, channels, image_size, filter_count,
      filter_size, pool_size_1, pool_size_2):
    self.batch_size = batch_size
    self.channels = channels
    self.image_size = image_size
    self.filter_count = filter_count
    self.filter_size = filter_size
    self.pool_size_1 = pool_size_1
    self.pool_size_2 = pool_size_2

    glorot = init.GlorotUniform()
    zero = init.Constant()
    self.w_conv1 = theano.shared(glorot.sample((filter_count, channels,
      filter_size, filter_size)))
    self.b_conv1 = theano.shared(zero.sample((filter_count,)))
    self.w_conv2 = theano.shared(glorot.sample((filter_count, filter_count,
      filter_size, filter_size)))
    self.b_conv2 = theano.shared(zero.sample((filter_count,)))

  def instantiate(self, *inputs):
    #print
    #print "inputs are", [i.l_output.output_shape for i in inputs]
    #print "my sizes are", (self.batch_size, self.channels, self.image_size)

    if len(inputs) == 0:
      l_first = layers.InputLayer((self.batch_size, self.channels, self.image_size, self.image_size))
      l_inputs = [l_first]
    else:
      l_first = layers.ConcatLayer([input.l_output for input in inputs], axis=1)
      l_inputs = [l_input for input in inputs for l_input in input.l_inputs]

    l_conv1 = layers.Conv2DLayer(l_first, self.filter_count, self.filter_size,
        W=self.w_conv1, b=self.b_conv1, border_mode="same")
    if self.pool_size_1 > 1:
      l_pool1 = layers.MaxPool2DLayer(l_conv1, self.pool_size_1)
      l_1 = l_pool1
    else:
      l_1 = l_conv1
    l_drop1 = layers.DropoutLayer(l_1, p=0)

    l_conv2 = layers.Conv2DLayer(l_drop1, self.filter_count, self.filter_size,
        W=self.w_conv2, b=self.b_conv2, border_mode="same")
    if self.pool_size_2 > 1:
      l_pool2 = layers.MaxPool2DLayer(l_conv2, self.pool_size_2)
      l_2 = l_pool2
    else:
      l_2 = l_conv2
    l_drop2 = layers.DropoutLayer(l_2, p=0)

    l_output = l_drop2

    return Network(l_inputs, l_output)

class MLPModule:

  def __init__(self, batch_size, input_size, hidden_size, output_size,
      output_nonlinearity=None):
    self.batch_size = batch_size
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.output_nonlinearity = output_nonlinearity

    glorot = init.GlorotUniform()
    zero = init.Constant()
    self.w_hidden = theano.shared(glorot.sample((input_size, hidden_size)))
    self.b_hidden = theano.shared(zero.sample((hidden_size,)))
    self.w_output = theano.shared(glorot.sample((hidden_size, output_size)))
    self.b_output = theano.shared(zero.sample((output_size,)))

  def instantiate(self, *inputs):
    #print
    #print "inputs are", [i.l_output.output_shape for i in inputs]
    #print "my sizes are", (self.batch_size, self.input_size)

    if len(inputs) == 0:
      l_first = layers.InputLayer((self.batch_size, self.input_size))
      l_inputs = [l_first]
    else:
      l_first = layers.ConcatLayer([input.l_output for input in inputs])
      l_inputs = [l_input for input in inputs for l_input in input.l_inputs]

    l_hidden = layers.DenseLayer(l_first, self.hidden_size, W=self.w_hidden,
            b=self.b_hidden)
            
    l_output = layers.DenseLayer(l_hidden, self.output_size, W=self.w_output,
            b=self.b_output, nonlinearity=self.output_nonlinearity)

    return Network(l_inputs, l_output)

class NMN:
  def __init__(self, input_type, output_type, params):
    self.cached_networks = dict()
    self.input_type = input_type
    self.output_type = output_type

    self.modules = dict()
    for key, mod_config in params["modules"].items():
      module = eval(mod_config)
      self.modules[key] = module

  def get_network(self, query):
    if query in self.cached_networks:
      return self.cached_networks[query]

    logging.debug('new network: %s', pp(query))
    pre_net = wire(query, self.modules)
    net = self.modules['_output'].instantiate(pre_net)
    net.add_objective(self.input_type, self.output_type)

    self.cached_networks[query] = net
    return net

  def train(self, query, input_, output):
    network = self.get_network(query)
    return network.train(input_, output)

  def loss(self, query, input_, output):
    network = self.get_network(query)
    return network.loss(input_, output)

  def predict(self, query, input_):
    network = self.get_network(query)
    return network.predict(input_)
