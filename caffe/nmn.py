#!/usr/bin/env python2

import itertools

class NMNModel:
    def __init__(self, config):
        self.config = config

    def train(self, train_data, val_data):
        train_by_query = dict(itertools.groupby(train_data, lambda d: d.query))
        val_by_query = dict(itertools.groupby(val_data, lambda d: d.query))
