#!/usr/bin/env python2

from collections import namedtuple

class Layout:
    def __init__(self, modules, indices):
        assert isinstance(modules, tuple)
        assert isinstance(indices, tuple)
        self.modules = modules
        self.indices = indices

    def __eq__(self, other):
        return isinstance(other, Layout) and \
                other.modules == self.modules and \
                other.indices == self.indices

    def __hash__(self):
        return hash(self.modules) + 3 * hash(self.indices)

    def __str__(self):
        return self.__str_helper(self.modules, self.indices)

    def __str_helper(self, modules, indices):
        if isinstance(modules, tuple):
            mhead, mtail = modules[0], modules[1:]
            ihead, itail = indices[0], indices[1:]
            mod_name = mhead.__name__
            below = [self.__str_helper(m, i) for m, i in zip(mtail, itail)]
            return "(%s[%s] %s)" % (mod_name, ihead, " ".join(below))

        return "%s[%s]" % (modules.__name__, indices)

class Datum:
    def __init__(self):
        self.id = None
        self.string = None
        self.layout = None
        self.outputs = None

    def load_input(self):
        raise NotImplementedError()
