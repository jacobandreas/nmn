#!/usr/bin/env python2

from nmn import NMNModel

def build_model(config):
    if config.name == "nmn":
        return NMNModel(config)
    else:
        raise NotImplementedError(
                "Don't know how to build a %s model" % config.name)
