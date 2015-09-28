#!/usr/bin/env python2

from images import ImageTask

def load_task(config):
    if config.name == "images":
        return ImageTask(config)
    else:
        raise NotImplementedError(
                "Don't know how to build a %s task" % config.name)
