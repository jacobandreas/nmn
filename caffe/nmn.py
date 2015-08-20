#!/usr/bin/env python2

import itertools
import importlib

class ConvModule:
    def __init__(self):
        pass

class ModuleNetwork:
    def __init__(self, query, model):
        module_ordering = []
        child_indices = []
        self.build_module_ordering(module_ordering, child_indices, query, model)

    def build_module_ordering(self, module_ordering, child_indices, query, 
                              model):
        if not isinstance(query, tuple):
            module = model.get_module(query, 0)
            module_ordering.append(module)
            child_indices.append(())
        else:
            head = query[0]
            tail = query[1:]
            arity = len(tail)
            module = model.get_module(query, arity)
            my_index = len(module_ordering)
            module_ordering.append(module)
            child_indices.append([])
            for child in tail:
                child_index = len(module_ordering)
                child_indices[my_index].append(child_index)
                self.build_module_ordering(module_ordering, child_indices,
                                           child, model)
            child_indices[my_index] = tuple(child_indices[my_index])

    def forward(self, datum):
        pass

    def backward(self, datum):
        pass

class NMNModel:
    def __init__(self, config):
        self.config = config
        self.module_builder = importlib.import_module(
                "task.%s" % config.module_builder)
        self.modules = dict()

    def compute(self, query, data, include_grad=False):
        ModuleNetwork(query, self)

    def train(self, train_data, val_data):
        train_by_query = dict(itertools.groupby(train_data, lambda d: d.query))
        val_by_query = dict(itertools.groupby(val_data, lambda d: d.query))

        for query, data in train_by_query.items():
            self.compute(query, data, include_grad=True)

    def get_module(self, name, arity):
        if (name, arity) not in self.modules:
            module = self.module_builder.build_caffe_module(name, arity)
        return None
