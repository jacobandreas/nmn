#!/usr/bin/env python2

import nmn

from lasagne import nonlinearities
from lasagne import nonlinearities, layers

class ImagesModuleBuilder:
  def __init__(self, params):
    self.channels = params["channels"]
    self.vocab_size = params["vocab_size"]
    self.meta_module = nmn.ConvWithEmbeddingMetaModule(256, 64)

  def build_input(self):
    return layers.InputLayer((None, self.channels, None, None))

  def build(self, name, arity):
    if name == "_output":
      return nmn.AttentionModule(256, self.vocab_size)

    elif name == "_pre":
      return nmn.InputIdentityModule()

    else:
      if arity == 0:
        return self.meta_module.get_module(name)
      else:
        return nmn.IdentityModule()
