#!/usr/bin/env python2

import numpy as np

class Rule:
  def __init__(self, lhs, rhs, weight):
    self.lhs = lhs
    self.rhs = rhs
    self.weight = weight

def sample(symbol, grammar):
  # pick a rule
  rules = grammar[symbol]
  exps = [rule.rhs for rule in rules]
  weights = [rule.weight for rule in rules]
  index = np.random.choice(len(exps), p=weights)
  exp = exps[index]

  # generate recursively
  exp = list(exp)
  for i in range(len(exp)):
    sym = exp[i]
    if isinstance(sym, str) and sym[0] == '$':
      exp[i] = sample(sym, grammar)
    else:
      exp[i] = sym

  if len(exp) == 1:
    return exp[0]
  else:
    return tuple(exp)

def pp(exp):
  if isinstance(exp, tuple):
    return "(%s)" % (" ".join([pp(e) for e in exp]))
  return str(exp)

