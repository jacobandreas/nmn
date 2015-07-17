#!/usr/bin/env python2

from data.util import pp
import logging
from lasagne import init, layers, objectives, updates
import numpy as np
import theano
import theano.tensor as T
import warnings

class Network:
  def __init__(self, l_inputs, l_output, t_input, t_output, t_target, t_loss, params):
    self.l_inputs = l_inputs
    self.l_output = l_output

    momentum = 0.9
    lr = .0001
    upd = updates.momentum(t_loss, params, lr, momentum)

    loss_inputs = [t_input, t_target]
    self.loss = theano.function(loss_inputs, t_loss)
    self.train = theano.function(loss_inputs, t_loss, updates=upd)

def wire(query, modules):
  if not isinstance(query, tuple):
    return modules.get(query).instantiate()
  args = [wire(q, modules) for q in query[1:]]
  return modules.get(query[0]).instantiate(*args)

class MLPModule:

  def __init__(self, batch_size, input_size, hidden_size, output_size):
    self.batch_size = batch_size
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    glorot = init.GlorotUniform()
    self.w_hidden = theano.shared(glorot.sample((input_size, hidden_size)))
    self.w_output = theano.shared(glorot.sample((hidden_size, output_size)))

  def instantiate(self, *inputs):
    if len(inputs) == 0:
      l_first = layers.InputLayer((self.batch_size, self.input_size))
      l_inputs = [l_first]
    else:
      l_first = layers.ConcatLayer([input.l_output for input in inputs])
      l_inputs = [l_input for input in inputs for l_input in input.l_inputs]

    l_hidden = layers.DenseLayer(l_first, self.hidden_size, W=self.w_hidden)
    l_output = layers.DenseLayer(l_hidden, self.output_size, W=self.w_output, nonlinearity=None)

    t_input = T.matrix("input")
    input_mapping = {l_input: t_input for l_input in l_inputs}
    t_output = layers.get_output(l_output, input_mapping)
    t_target = T.matrix("target")
    t_loss = T.mean(objectives.squared_error(t_output, t_target))

    params = layers.get_all_params(l_output)

    return Network(l_inputs, l_output, t_input, t_output, t_target, t_loss, params)

class NMN:
  def __init__(self, params):
    self.cached_networks = dict()

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

    self.cached_networks[query] = net
    return net

  def train(self, query, input_, output):
    network = self.get_network(query)
    return network.train(input_, output)

  def loss(self, query, input_, output):
    network = self.get_network(query)
    return network.loss(input_, output)
