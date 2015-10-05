#!/usr/bin/env python2

from images import ImageTask
from daquar import DaquarTask
from cocoqa import CocoQATask

def load_task(config):
    if config.name == "images":
        return ImageTask(config)
    elif config.name == "daquar":
        return DaquarTask(config)
    elif config.name == "cocoqa":
        return CocoQATask(config)
    else:
        raise NotImplementedError(
                "Don't know how to build a %s task" % config.name)
