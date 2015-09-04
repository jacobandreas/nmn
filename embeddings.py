#!/usr/bin/env python2

import numpy as np
import cPickle as pickle

def load(path):
    matrix = np.load(path + ".data.npy")
    with open(path + ".index.pkl", "rb") as index_f:
        index = pickle.load(index_f)
    return matrix, index
