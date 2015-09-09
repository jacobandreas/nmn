#!/usr/bin/env python2

from data.util import pp
from shapes import *
from images import *
from util import Index

import logging
from lasagne import init, layers, objectives, updates, nonlinearities
#import lasagne.layers.cuda_convnet
import numpy as np
import theano
import theano.tensor as T
import warnings

class Network:
  def __init__(self, l_input, l_output, l_diagnose=None):
    self.l_input = l_input
    self.l_output = l_output
    self.l_diagnose = l_diagnose

  def add_objective(self, input_type, output_type):
    params = layers.get_all_params(self.l_output)
    self.params = params

    t_input = input_type.make_input()
    t_output = layers.get_output(self.l_output, t_input)
    t_output_det = layers.get_output(self.l_output, t_input, deterministic=True)
    t_pred = T.argmax(t_output_det, axis=1)
    t_target = output_type.make_target()

    t_diagnose = layers.get_output(self.l_diagnose, t_input, deterministic=True)

    lr = 0.001
    clip = 1.0

    t_loss = objectives.aggregate(output_type.loss(t_output, t_target))
    grads = T.grad(t_loss, params)
    scaled_grads, t_norm = updates.total_norm_constraint(grads, clip, return_norm=True)
    #upd = updates.adadelta(scaled_grads, params, learning_rate=lr)
    #upd = updates.adam(scaled_grads, params, learning_rate = lr)
    upd = updates.nesterov_momentum(scaled_grads, params, learning_rate=lr)
    #upd = updates.momentum(scaled_grads, params, learning_rate=lr)
    #upd = updates.sgd(scaled_grads, params, learning_rate = lr)

    loss_inputs = [t_input, t_target]
    self.loss = theano.function(loss_inputs, t_loss)
    self.train = theano.function(loss_inputs, t_loss, updates=upd,
    )#mode="DebugMode")
    self.predict = theano.function([t_input], t_pred)
    self.response = theano.function([t_input], t_output_det)
    self.diagnose = theano.function([t_input], t_diagnose)
    self.norm = theano.function(loss_inputs, t_norm)
    self.grad = theano.function(loss_inputs, grads)

class IdentityModule:
  def instantiate(self, input, *below):
    assert len(below) == 1
    return below[0]

  def write_weights(self, dest, name):
    pass

class InputIdentityModule:
  def instantiate(self, input, *below):
    assert len(below) == 0
    return Network(input, input)

  def write_weights(self, dest, name):
    pass

class MinModule:
  def __init__(self):
    pass

  def instantiate(self, l_input, *below):
    l_min = layers.ElemwiseMergeLayer([b.l_output for b in below], T.minimum)
    l_threshold = layers.NonlinearityLayer(l_min)
    l_output = l_threshold
    return Network(l_input, l_output)

  def write_weights(self, dest, name):
    pass

class AppendConstantLayer(layers.Layer):
  def __init__(self, incoming, vocab_size, embedding_size, embeddings, index, **kwargs):
    super(AppendConstantLayer, self).__init__(incoming, **kwargs)
    self.embedding_size = embedding_size
    self.embedding = embeddings[index,:]
    self.e_shaped = T.reshape(self.embedding, (embedding_size, 1, 1, 1))
    #self.add_param(embeddings, (vocab_size, embedding_size), name="embeddings",
    #    trainable=False)
    #self.batch_size = incoming.input_var.shape[0]
    #self.width = incoming.input_var.shape[2]
    #self.height = incoming.input_var.shape[3]

  def get_output_for(self, input, **kwargs):
    #batch_size, width, height = input.shape[0:3]
    #e_tiled = T.tile(self.e_shaped, (self.batch_size, self.width, self.height, 1))
    #print self.batch_size, self.width, self.height
    batch_size = input.shape[0]
    width = input.shape[2]
    height = input.shape[3]
    e_tiled = self.e_shaped * T.ones((1, batch_size, width, height))
    #e_shuffled = e_tiled.dimshuffle((0, 3, 1, 2))
    e_shuffled = e_tiled.dimshuffle((1, 0, 2, 3))
    #return e_shuffled
    embedded = e_shuffled
    return T.concatenate([embedded, input], axis=1)

  def get_output_shape_for(self, input_shape):
    return (input_shape[0], 
            input_shape[1] + self.embedding_size,
            input_shape[2],
            input_shape[3])

class ConvWithEmbeddingMetaModule:
  EMBEDDING_SIZE = 100
  EMBEDDING_PATH = "data/embeddings/skipdep_embeddings_lc.txt"

  def __init__(self, channels, hidden_size):
    self.hidden_size = hidden_size

    glorot = init.GlorotUniform()
    zero = init.Constant()

    #self.w_hidden = theano.shared(glorot.sample((channels + self.EMBEDDING_SIZE,
    #  hidden_size)))
    #self.b_hidden = theano.shared(zero.sample((hidden_size,)))

    #self.w_hidden = theano.shared(np.ones((channels + self.EMBEDDING_SIZE,
    #    hidden_size), dtype=theano.config.floatX))
    #self.b_hidden = theano.shared(np.zeros((hidden_size,),
    #    dtype=theano.config.floatX))

    self.w_hidden = theano.shared(np.load("weights/conv_with_embedding.w_hidden.npy"))
    self.b_hidden = theano.shared(np.load("weights/conv_with_embedding.b_hidden.npy"))

    #self.w_out = theano.shared(glorot.sample((hidden_size, 1)))
    #self.b_out = theano.shared(zero.sample((1,)))

    #self.w_out = theano.shared(np.ones((hidden_size, 1), dtype=theano.config.floatX))
    #self.b_out = theano.shared(np.zeros((1,), dtype=theano.config.floatX))

    self.w_out = theano.shared(np.load("weights/conv_with_embedding.w_out.npy"))
    self.b_out = theano.shared(np.load("weights/conv_with_embedding.b_out.npy"))

    self.index, self.embeddings = self.load_embeddings()
    self.vocab_size = len(self.index)

  def load_embeddings(self):
    index = Index()
    vecs = []
    with open(self.EMBEDDING_PATH) as embedding_file:
      for line in embedding_file:
        word, comps = line.split("\t")
        vec = [float(f) for f in comps.split()]
        assert len(vec) == self.EMBEDDING_SIZE
        index.index(word)
        vecs.append(vec)

    return index, theano.shared(np.asarray(vecs, dtype=theano.config.floatX))

  def get_module(self, name):
    class ConvInstanceModule:
      def __init__(self, outer):
        self.outer = outer
      def instantiate(self, l_input, *below):
        assert len(below) == 1
        word_index = self.outer.index[name]
        if word_index is None:
          word_index = self.outer.index["*unknown*"]
          logging.warn("unknown word: %s", name)
          assert word_index is not None
        #l_drop = layers.DropoutLayer(below[0].l_output)
        l_appended = AppendConstantLayer(below[0].l_output,
                                         #l_drop,
                                         self.outer.vocab_size,
                                         self.outer.EMBEDDING_SIZE,
                                         self.outer.embeddings,
                                         word_index)
        l_hidden = layers.NINLayer(l_appended, self.outer.hidden_size,
                                   W=self.outer.w_hidden, b=self.outer.b_hidden)
        l_output = layers.NINLayer(l_hidden, 1, W=self.outer.w_out,
                                   b=self.outer.b_out)

        l_diagnose = l_output

        return Network(l_input, l_output, l_diagnose)
      def write_weights(self, dest, name):
        np.save(dest + "/conv_with_embedding.w_hidden", self.outer.w_hidden.get_value())
        np.save(dest + "/conv_with_embedding.b_hidden", self.outer.b_hidden.get_value())
        np.save(dest + "/conv_with_embedding.w_out", self.outer.w_out.get_value())
        np.save(dest + "/conv_with_embedding.b_out", self.outer.b_out.get_value())

    return ConvInstanceModule(self)

class Conv1Module:
  def __init__(self, batch_size, channels, image_size, filter_count_1,
      filter_size_1, pool_size_1):
    self.batch_size = batch_size
    self.channels = channels
    self.image_size = image_size
    self.filter_count_1 = filter_count_1
    self.filter_size_1 = filter_size_1
    self.pool_size_1 = pool_size_1

    glorot = init.GlorotUniform()
    zero = init.Constant()
    self.w_conv1 = theano.shared(glorot.sample((filter_count_1, channels,
      filter_size_1, filter_size_1)))
    self.b_conv1 = theano.shared(zero.sample((filter_count_1,)))

  def instantiate(self, l_input, *below):

    if len(below) == 0:
      l_first = layers.InputLayer((self.batch_size, self.channels, self.image_size, self.image_size))
    else:
      l_first = layers.ConcatLayer([b.l_output for b in below], axis=1)

    l_conv1 = layers.Conv2DLayer(l_first, self.filter_count_1,
        self.filter_size_1,
        W=self.w_conv1, b=self.b_conv1, border_mode="same")
    if self.pool_size_1 > 1:
      l_pool1 = layers.MaxPool2DLayer(l_conv1, self.pool_size_1)
      l_1 = l_pool1
    else:
      l_1 = l_conv1

    l_output = l_1

    return Network(l_input, l_output)

  def write_weights(self, dest, name):
    np.save(dest + "/" + name, self.w_conv1.get_value())

class ConvModule:
  def __init__(self, batch_size, channels, image_size, filter_count_1,
      filter_count_2, filter_size_1, filter_size_2, pool_size_1, pool_size_2,
      tie=True):
    self.batch_size = batch_size
    self.channels = channels
    self.image_size = image_size
    self.filter_count_1 = filter_count_1
    self.filter_count_2 = filter_count_2
    self.filter_size_1 = filter_size_1
    self.filter_size_2 = filter_size_2
    self.pool_size_1 = pool_size_1
    self.pool_size_2 = pool_size_2
    self.tie = tie

    glorot = init.GlorotUniform()
    zero = init.Constant()
    if tie:
      self.w_conv1 = theano.shared(glorot.sample((filter_count_1, channels,
        filter_size_1, filter_size_1)))
      self.b_conv1 = theano.shared(zero.sample((filter_count_1,)))
      self.w_conv2 = theano.shared(glorot.sample((filter_count_2, filter_count_1,
        filter_size_2, filter_size_2)))
      self.b_conv2 = theano.shared(zero.sample((filter_count_2,)))
    else:
      self.w_conv1 = glorot
      self.w_conv2 = glorot
      self.b_conv1 = zero
      self.b_conv2 = zero

  def instantiate(self, l_input, *below):

    if len(below) == 0:
      #l_first = layers.InputLayer((self.batch_size, self.channels, self.image_size, self.image_size))
      assert False
    else:
      l_first = layers.ConcatLayer([b.l_output for b in below], axis=1)

    l_conv1 = layers.Conv2DLayer(l_first, self.filter_count_1,
        self.filter_size_1,
        W=self.w_conv1, b=self.b_conv1, border_mode="same")
    if self.pool_size_1 > 1:
      l_pool1 = layers.MaxPool2DLayer(l_conv1, self.pool_size_1)
      l_1 = l_pool1
    else:
      l_1 = l_conv1

    l_conv2 = layers.Conv2DLayer(l_1, self.filter_count_2, self.filter_size_2,
        W=self.w_conv2, b=self.b_conv2, border_mode="same")
    if self.pool_size_2 > 1:
      l_pool2 = layers.MaxPool2DLayer(l_conv2, self.pool_size_2)
      l_2 = l_pool2
    else:
      l_2 = l_conv2

    l_output = l_2

    return Network(l_input, l_output)

  def write_weights(self, dest, name):
    if self.tie:
      np.save(dest + "/" + name, self.w_conv1.get_value())

class SumLayer(layers.Layer):
  def get_output_for(self, input, **kwargs):
    return input.sum(axis=(-1, -2))

  def get_output_shape_for(self, input_shape):
    return input_shape[:-2]

class ImageSoftmaxLayer(layers.Layer):
  def get_output_for(self, input, **kwargs):
    sh = input.shape
    shaped = input.reshape((sh[0], sh[2] * sh[3]))
    softmax = T.nnet.softmax(shaped)
    unshaped = softmax.reshape((sh[0], 1, sh[2], sh[3]))
    return unshaped

class AttentionLayer(layers.MergeLayer):
  def __init__(self, incoming_data, incoming_attention):
    super(AttentionLayer, self).__init__([incoming_data, incoming_attention])

  def get_output_for(self, inputs, **kwargs):
    input_data, input_attention = inputs
    input_attention_bc = T.addbroadcast(input_attention, 1)
    attended = T.mul(input_data, input_attention_bc)
    collected = T.sum(attended, axis=(2,3))
    return collected

  def get_output_shape_for(self, input_shapes):
    return input_shapes[0][0:2]

class AttentionModule:
  def __init__(self, channels, output_size):
    self.output_size = output_size

    glorot = init.GlorotUniform()
    zero = init.Constant()
    #print self.w.get_value()
    #print np.sum(self.w.get_value())
    #print np.sum(np.square(self.w.get_value()))
    #self.w = theano.shared(init.Constant(1.).sample((channels, output_size)))

    #self.w = theano.shared(glorot.sample((channels, output_size)))
    #self.b = theano.shared(zero.sample((output_size,)))

    self.w = theano.shared(np.load("weights/_output-1.w.npy"))
    self.b = theano.shared(np.load("weights/_output-1.b.npy"))

  def instantiate(self, l_input, *below):
    assert len(below) == 1
    assert below[0].l_output.output_shape[1] == 1
    l_sm = ImageSoftmaxLayer(below[0].l_output)
    l_att = AttentionLayer(l_input, l_sm)
    l_dense = layers.DenseLayer(l_att, self.output_size, W=self.w, b=self.b,
                                nonlinearity=nonlinearities.softmax)

    return Network(l_input, l_dense, l_diagnose=below[0].l_diagnose)
    #return Network(l_input, l_att)

  def write_weights(self, dest, name):
    np.save(dest + "/" + name + ".w", self.w.get_value())
    np.save(dest + "/" + name + ".b", self.b.get_value())

class MLPModule:

  def __init__(self, batch_size, input_size, hidden_size, output_size,
      output_nonlinearity=nonlinearities.rectify):
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

  def instantiate(self, l_input, *below):

    if len(below) == 0:
      #l_first = layers.InputLayer((self.batch_size, self.input_size))
      assert False
    else:
      l_first = layers.ConcatLayer([b.l_output for b in below])

    l_hidden = layers.DenseLayer(l_first, self.hidden_size, W=self.w_hidden,
            b=self.b_hidden)
            
    l_output = layers.DenseLayer(l_hidden, self.output_size, W=self.w_output,
            b=self.b_output, nonlinearity=self.output_nonlinearity)

    if self.output_nonlinearity != nonlinearities.softmax:
      l_shape = layers.ReshapeLayer(l_output, (self.batch_size, 1, 3, 3))
      l_out = l_shape
    else:
      l_out = l_output

    return Network(l_input, l_out)

  #def write_weights(self, dest, name):
  #  pass

class NMN:
  def __init__(self, input_type, output_type, params):
    self.cached_networks = dict()
    self.input_type = input_type
    self.output_type = output_type

    self.modules = dict()
    #if "modules" in params:
    #  for key, mod_config in params["modules"].items():
    #    module = eval(mod_config)
    #    self.modules[key] = module

    self.module_builder = globals()[params["module_builder"]["class"]](params["module_builder"])

  def wire(self, query, l_input=None, l_pre=None):
    if l_input is None:
      assert l_pre is None
      l_input = self.get_input()
      l_pre = self.get_module("_pre", -1).instantiate(l_input)
    if not isinstance(query, tuple):
      return self.get_module(query, 0).instantiate(l_input, l_pre)
    args = [self.wire(q, l_input, l_pre) for q in query[1:]]
    return self.get_module(query[0], len(args)).instantiate(l_input, *args)

  def get_input(self):
    return self.module_builder.build_input()

  def get_module(self, name, arity):
    if (name, arity) in self.modules:
      return self.modules[name, arity]
    module = self.module_builder.build(name, arity)
    self.modules[name, arity] = module
    return module

  def get_network(self, query):
    if query in self.cached_networks:
      return self.cached_networks[query]

    logging.debug('new network: %s', pp(query))
    pre_net = self.wire(query)
    net = self.get_module("_output", 1).instantiate(pre_net.l_input, pre_net)
    net.add_objective(self.input_type, self.output_type)

    self.cached_networks[query] = net
    return net

  def train(self, query, input_, output, return_norm=True):
    network = self.get_network(query)
    loss = network.train(input_, output)
    if return_norm:
      norm = network.norm(input_, output)
      return loss, norm
    else:
      return loss

  def loss(self, query, input_, output):
    network = self.get_network(query)
    return network.loss(input_, output)

  def predict(self, query, input_):
    network = self.get_network(query)
    return network.predict(input_)

  def response(self, query, input_):
    network = self.get_network(query)
    return network.response(input_)

  def grad(self, query, input, output):
    network = self.get_network(query)
    return network.grad(input, output)

  def diagnose(self, query, input):
    network = self.get_network(query)
    return network.diagnose(input)

  def serialize(self, dest):
    for name_and_arity, module in self.modules.items():
      module.write_weights("weights", "%s-%s" % name_and_arity)
