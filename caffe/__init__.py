#!/usr/bin/env python2

from nmn import NMNModel

def build_model(config, opt_config):
    if config.name == "nmn":
        return NMNModel(config, opt_config)
    else:
        raise NotImplementedError(
                "Don't know how to build a %s model" % config.name)
