#!/usr/bin/env python2

import nmn
from lasagne import nonlinearities
from lasagne import nonlinearities, layers

class ImagesModuleBuilder:
  def __init__(self, params):
    self.batch_size = params["batch_size"]
    self.image_size = params["image_size"]
    self.channels = params["channels"]
    self.vocab_size = params["vocab_size"]

  def build_input(self):
    return layers.InputLayer((self.batch_size, self.channels, self.image_size,
      self.image_size))

  def build(self, name, arity):
    if name == "_output":
      return nmn.MLPModule(self.batch_size, self.image_size * self.image_size *
          self.channels, 256, self.vocab_size,
          output_nonlinearity=nonlinearities.softmax)

    elif name == "_pre":
      return nmn.InputIdentityModule()

    else:
      return nmn.IdentityModule()
