#!/usr/bin/env python2

import cPickle as pickle
import numpy as np
import sys

WORD_LIMIT = 10000

vecs = []
words = []
for i, line in enumerate(sys.stdin):
    word, svec = line.split("\t")
    vec = np.asarray([float(v) for v in svec.split()])
    vecs.append(vec)
    words.append(word)
    if i == WORD_LIMIT - 1:
        break

mat = np.asarray(vecs)

word_index = dict()
for word in words:
    word_index[word] = len(word_index)

np.save(sys.argv[1] + ".data", mat)
with open(sys.argv[1] + ".index.pkl", "w") as index_f:
    pickle.dump(word_index, index_f)
