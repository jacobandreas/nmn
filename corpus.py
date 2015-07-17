#!/usr/bin/env python2

import numpy as np
import sexpdata

class QueryDatum:
  def __init__(self, query, input_, output):
    self.query = query
    self.input_ = input_
    self.output = output

def extract_query(sexp_query):
  if isinstance(sexp_query, sexpdata.Symbol):
    return sexp_query.value()
  return tuple(extract_query(q) for q in sexp_query)

def parse_query(query):
  parsed = sexpdata.loads(query)
  extracted = extract_query(parsed)
  return extracted

def load_math(set_name):
  data = []
  with open("data/math/%s.query" % set_name) as query_f, \
       open("data/math/%s.input" % set_name) as input_f, \
       open("data/math/%s.output" % set_name) as output_f:
    for query_str, input_str, output_str in zip(query_f, input_f, output_f):
      query = parse_query(query_str.strip())
      input_ = np.asarray([float(f) for f in input_str.strip().split()])
      output = np.asarray([float(f) for f in output_str.strip().split()])
      data.append(QueryDatum(query, input_, output))
  return data

def load(corpus_name, set_name):
  if corpus_name == 'math':
    return load_math(set_name)
  else:
    assert False

if __name__ == '__main__':
  load_math("train.tiny")
