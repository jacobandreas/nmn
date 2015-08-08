#!/usr/bin/env python2

import corpus
import nmn as mod_nmn
from input_types import ImageInput
from output_types import ImageOutput

import numpy as np
import theano
import theano.tensor as T
from lasagne import layers, nonlinearities
import sys

class HackColorModule:
  def __init__(self, channel):
    conv_weights = np.zeros((1, 3, 5, 5), dtype=theano.config.floatX)
    #conv_weights[0,:,:,:] = -3
    #conv_weights[0,channel,:,:] = 1
    conv_weights[0,:,:,:] += 0.01 * np.random.random((3,5,5))
    self.w_conv1 = theano.shared(conv_weights)
    self.b_conv1 = theano.shared(-0 * np.ones((1,), dtype=theano.config.floatX))

  def instantiate(self, *inputs):
    l_first = layers.InputLayer((256, 3, 30, 30))
    l_inputs = [l_first]

    l_conv = layers.Conv2DLayer(l_first, 1, 5, W=self.w_conv1,
        b=self.b_conv1, border_mode="same")
    l_pool = layers.MaxPool2DLayer(l_conv, 10)
    l_output = l_pool

    #l_shape = layers.ReshapeLayer(l_pool, (256,9))
    #l_output = l_shape

    return mod_nmn.Network(l_inputs, l_output)

  def write_weights(self, dest, name):
    np.save(dest + "/" + name, self.w_conv1.get_value())

class HackShapeModule:
  def __init__(self, channel):
    conv_weights = np.zeros((1, 3, 5, 5), dtype=theano.config.floatX)
    if channel == 0:
      #conv_weights[0,:,:,:] = np.asarray([[
      #  [0., 0., 0., 0., 0.],
      #  [0., -1., -1., -1., -1.],
      #  [0., -1., 1., 1., 1.],
      #  [0., -1., 1., 0., 0.],
      #  [0., -1., 1., 0., 0.]
      #]])
      conv_weights[0,:,:,:] = np.asarray([[
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 1., -1.],
        [0., 0., 1., -1., 0.],
        [0., 1., -1., 0., 0.],
        [1., -1., 0., 0., 0.]
      ]])
    elif channel == 1:
      pass
      #conv_weights[0,:,:,:] = np.asarray([[
      #  [0., 0., 0., 0., 1.],
      #  [0., 0., 0., 0., 1.],
      #  [0., 0., 0., 1., 1.],
      #  [0., 0., 0., 1., 1.],
      #  [0., 0., 1., 1., 1.]
      #]])
    #conv_weights -= 0.9
    #conv_weights += 0.1 * np.random.random((3,5,5))

    self.w_conv1 = theano.shared(conv_weights)
    self.b_conv1 = theano.shared(-0 * np.zeros((1,), dtype=theano.config.floatX))

  def instantiate(self, *inputs):
    l_first = layers.InputLayer((256, 3, 30, 30))
    l_inputs = [l_first]

    l_conv = layers.Conv2DLayer(l_first, 1, 5, W=self.w_conv1,
        b = self.b_conv1, border_mode="same")
    l_pool = layers.MaxPool2DLayer(l_conv, 10)
    l_output = l_pool

    return mod_nmn.Network(l_inputs, l_output)

  def write_weights(self, dest, name):
    np.save(dest + "/" + name, self.w_conv1.get_value())

class HackL1ConvModule:
  def __init__(self, channel):
    conv_weights = np.zeros((1, 6, 3, 3), dtype=theano.config.floatX)
    conv_weights[0,channel,1,1] = 1.
    self.w_conv1 = theano.shared(conv_weights)
    self.b_conv1 = theano.shared(np.zeros((1,), dtype=theano.config.floatX))

  def instantiate(self, *inputs):
    l_first = layers.InputLayer((256, 6, 3, 3))
    l_inputs = [l_first]

    l_conv = layers.Conv2DLayer(l_first, 1, 3, W=self.w_conv1,
        b=self.b_conv1, border_mode="same")
    l_output = l_conv

    return mod_nmn.Network(l_inputs, l_output)

class HackL2ConvModule:
  def __init__(self, r, c):
    conv_weights = np.zeros((1, 1, 3, 3), dtype=theano.config.floatX)
    conv_weights[0,0,r,c] = 1.
    self.w_conv1 = theano.shared(conv_weights)
    self.b_conv1 = theano.shared(np.zeros((1,), dtype=theano.config.floatX))

  def instantiate(self, *inputs):
    l_first = layers.ConcatLayer([input.l_output for input in inputs])
    l_inputs = [l_input for input in inputs for l_input in input.l_inputs]

    l_conv = layers.Conv2DLayer(l_first, 1, 3, W=self.w_conv1,
        b=self.b_conv1, border_mode="same")
    l_output = l_conv

    return mod_nmn.Network(l_inputs, l_output)

  def write_weights(self, dest, name):
    np.save(dest + "/" + name, self.w_conv1.get_value())

class HackL3Module:
  def __init__(self):
    pass

  def instantiate(self, *inputs):
    l_first = layers.ElemwiseMergeLayer([input.l_output for input in inputs],
        theano.tensor.minimum)
    l_threshold = layers.NonlinearityLayer(l_first)
    l_inputs = [l_input for input in inputs for l_input in input.l_inputs]
    l_output = l_threshold
    return mod_nmn.Network(l_inputs, l_output)

  def write_weights(self, dest, name):
    pass

if __name__ == "__main__":
  data = corpus.load("shapes", "val")
  np.random.shuffle(data)

  nmn = mod_nmn.NMN(ImageInput(), ImageOutput(), {})
  nmn.modules["square"] = HackShapeModule(0)
  #nmn.modules["left_of"] = HackL2ConvModule(1, 0)
  nmn.modules["_output"] = mod_nmn.IdentityModule()

  net = nmn.get_network("square")

  l_input = layers.InputLayer((1, 3, 30, 30))
  #l_conv = layers.Conv2DLayer(l_input, 1, 3, border_mode="same")

  #t_input = T.tensor4("input")
  #t_output = layers.get_output(l_conv, t_input)
  #predict = theano.function([t_input], t_output)

  for datum in data:
    print "\n==\n"

    datum_inp = datum.input_.astype(theano.config.floatX)
    #for r in range(3):
    #  for c in range(3):
    #    if sum(inp[0:3,r,c]) < 0.5:
    #      print ".",
    #    else:
    #      shape = inp[0:3,r,c].argmax()
    #      print shape,
    #  print

    #print "--"

    #for r in range(3):
    #  for c in range(3):
    #    if sum(inp[3:6,r,c]) < 0.5:
    #      print ".",
    #    else:
    #      color = inp[3:6,r,c].argmax()
    #      print color,
    #  print
    for channel in range(3):
      for r in range(0, 30, 1):
        for c in range(0, 30, 1):
          if datum_inp[channel,r,c] < 0.5: sys.stdout.write(".")
          else: sys.stdout.write("#")
        print
      print


    inp = np.zeros((256, 3, 30, 30), dtype=theano.config.floatX)
    inp[0,:,:,:] = datum_inp
    pred = net.response(inp)[0,:,:,:]
    np.set_printoptions(precision=3, linewidth=200)
    print pred
    exit()
    #print inp
    #print pred
    #if pred.any(): exit()
