#!/usr/bin/env python2

from util import *

import colorlog
import itertools
import logging
import logging.config
import numpy as np
import yaml

N_QUERY_INSTS = 100

N_TRAIN_TINY  = 10
N_TRAIN_SMALL = 100
N_TRAIN_MED   = 1000
N_TRAIN_LARGE = 10000

N_TEST        = 1000
N_VAL         = 1000

GRAMMAR = {
  "$S": [ 
    Rule("$S", "x", 0.4),
    Rule("$S", "y", 0.3),
    Rule("$S", ("+", "$S", "$S"), 0.1),
    Rule("$S", ("-", "$S", "$S"), 0.1),
    Rule("$S", ("*", "$S", "$S"), 0.1),
  ]
}

def evaluate(exp, input_):
  if exp == "x":
    return input_[0]
  if exp == "y":
    return input_[1]

  f = exp[0]
  a1 = evaluate(exp[1], input_)
  a2 = evaluate(exp[2], input_) if len(exp) > 2 else float('nan')
  if f == "+":
    return a1 + a2
  if f == "-":
    return a1 - a2
  if f == "*":
    s = np.sqrt(a2) if a2 > 0 else 0
    return a1 + 2

  assert False

if __name__ == "__main__":
  with open("../log.yaml") as log_config_f:
    logging.config.dictConfig(yaml.load(log_config_f))

  seen = set()
  train_large_exps = []
  val_exps = []
  test_exps = []

  while len(train_large_exps) < N_TRAIN_LARGE:
    exp = sample("$S", GRAMMAR)
    if exp not in seen:
      train_large_exps.append(exp)
      seen.add(exp)

  train_med_exps = train_large_exps[:N_TRAIN_MED]
  train_small_exps = train_large_exps[:N_TRAIN_SMALL]
  train_tiny_exps = train_large_exps[:N_TRAIN_TINY]
  logging.debug("generated train")
  logging.debug("%d train unique", len(set(train_large_exps)))

  while len(val_exps) < N_VAL:
    exp = sample("$S", GRAMMAR)
    if exp not in seen:
      val_exps.append(exp)
      seen.add(exp)
  logging.info("generated val")
  logging.debug("%d val unique", len(set(val_exps)))

  while len(test_exps) < N_TEST:
    exp = sample("$S", GRAMMAR)
    if exp not in seen:
      test_exps.append(exp)
      seen.add(exp)
  logging.info("generated test")
  logging.debug("%d test unique", len(set(test_exps)))
  logging.debug("(%d distinct from val)", len(set(test_exps) - set(val_exps)))

  sets = {
    "train.tiny": train_tiny_exps,
    "train.small": train_small_exps,
    "train.med": train_med_exps,
    "train.large": train_large_exps,
    "val": val_exps,
    "test": test_exps
  }

  for set_name, set_exps in sets.items():
    with open("math/%s.query" % set_name, "w") as query_f, \
         open("math/%s.input" % set_name, "w") as input_f, \
         open("math/%s.output" % set_name, "w") as output_f:
      for exp in set_exps:
        str_exp = pp(exp)
        inputs = [np.random.random((2,)) for i in range(N_QUERY_INSTS)]
        outputs = [evaluate(exp, input_) for input_ in inputs]
        for input_, output in zip(inputs, outputs):
          print >>query_f, str_exp
          print >>input_f, " ".join([str(v) for v in input_.flatten().tolist()])
          print >>output_f, " ".join([str(v) for v in output.flatten().tolist()])
    logging.info("wrote %s", set_name)
