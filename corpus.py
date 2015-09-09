#!/usr/bin/env python2

from util import Index
from data.util import pp

from collections import defaultdict
import logging
import numpy as np
import sexpdata
import os

class QueryDatum:
  def __init__(self, query, input_, output, name=None):
    self.query = query
    self.input_ = input_
    self.output = output
    self.name = name

def extract_query(sexp_query):
  if isinstance(sexp_query, sexpdata.Symbol):
    return sexp_query.value()
  elif isinstance(sexp_query, int):
    return str(sexp_query)
  return tuple(extract_query(q) for q in sexp_query)

def parse_query(query):
  parsed = sexpdata.loads(query)
  extracted = extract_query(parsed)
  return extracted

def datum_filter(query, answer):
  if not isinstance(query, tuple) or query[0] != "color":
    return False
  if query not in (("color", "shirt"), ("color", "cat"), ("color", "train")):
    return False
  if " " in answer or "/" in answer or "," in answer:
    return False
  return True

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

def load_shapes(set_name):
  data = []
  index = {"true": 1, "false": 0}
  #index = {
  #  "1": 0,
  #  "2": 1,
  #  "3": 2,
  #  "4": 3,
  #  "5": 4,
  #  "6": 5,
  #  "7": 6,
  #  "8": 7,
  #  "9": 8,
  #  "red": 9,
  #  "green": 10,
  #  "blue": 11,
  #  "big": 12,
  #  "small": 13,
  #  "square": 14,
  #  "triangle": 15,
  #  "circle": 16
  #}
  with open("data/shapes/%s.query" % set_name) as query_f, \
       open("data/shapes/%s.output" % set_name) as output_f:
       #open("data/shapes/%s.input" % set_name) as input_f, \

    inputs = np.load("data/shapes/%s.input.npy" % set_name)
    for query_str, i_input, output_str in zip(query_f, range(inputs.shape[0]), output_f):
      query = parse_query(query_str.strip())
      #inp = np.asarray([float(f)/255. for f in input_str.strip().split()]).reshape((60,60,3)).transpose((2,0,1))
      inp = inputs[i_input,:,:,:].transpose((2,0,1)) / 255. - 0.5
      output = index[output_str.strip()]
      data.append(QueryDatum(query, inp, output))

  return data

def build_image_answer_index():
  index = Index()
  with open("data/images/processed/train.large.query") as query_f, \
       open("data/images/processed/train.large.output") as output_f:
    for qline, oline in zip(query_f, output_f):
      query = parse_query(qline.strip())
      answer = oline.strip()
      if not datum_filter(query, answer):
        continue
      index.index(answer)
  index.index("UNK")
  return index

def load_images(set_name):
  short_set_name = set_name.split(".")[0]
  data = []
  answer_index = build_image_answer_index()

  with open("data/images/processed/%s.query" % set_name) as query_f, \
       open("data/images/processed/%s.output" % set_name) as output_f, \
       open("data/images/processed/%s.answer_to_question" % set_name) as a2q_f, \
       open("data/images/processed/%s.question_to_image" % set_name) as q2i_f, \
       open("data/images/processed/%s.image_to_index" % set_name) as i2i_f:
    #inputs = np.load("data/images/processed/%s.input.npy" % set_name)

    image_names = os.listdir("data/images/Images/%s2014" % short_set_name)
    image_names = sorted(image_names)
    image_names = [n for n in image_names if n[-3:] == "jpg"]

    a2q = dict()
    q2i = dict()
    i2i = dict()

    question_ids = []
    answer_ids = []

    for line in a2q_f:
      answer, question = line.split(",")
      assert int(answer) not in a2q
      a2q[int(answer)] = int(question)
      answer_ids.append(int(answer))
    assert 0 not in a2q

    for line in q2i_f:
      question, image = line.split(",")
      assert int(question) not in q2i
      q2i[int(question)] = int(image)
      question_ids.append(int(question))
    assert 0 not in q2i

    for line in i2i_f:
      image, index = line.split(",")
      assert int(image) not in i2i
      i2i[int(image)] = int(index)

    assert len(image_names) == len(i2i)

    images = dict()
    for image_id, index in i2i.items()[:100]:
      try:
        images[image_id] = np.load("data/images/Images/%s2014/embedded/%s.npy" %
                (short_set_name, image_names[index]))
      except IOError as e:
        pass
        #logging.warn("couldn't find %s", index)

    questions = dict()
    for question_id, query in zip(question_ids, query_f):
      questions[question_id] = parse_query(query.strip())

    answer_counter = defaultdict(lambda: 0)
    for answer_id, answer in zip(answer_ids, output_f):
      question_id = a2q[answer_id]
      image_id = q2i[question_id]

      if image_id not in images:
        continue
      inp = images[image_id]
      query = questions[question_id]

      if not datum_filter(query, answer):
        continue

      output = answer_index[answer.strip()]
      if output is None:
        output = answer_index["UNK"]
      name = image_names[i2i[image_id]]
      datum = QueryDatum(query, inp, output, image_names[i2i[image_id]])
      data.append(datum)
      answer_counter[answer.strip()] += 1

      #print query, answer.strip(), output, name

      #break

  logging.info("%d items", len(data))
  #print answer_counter
  #exit()
  return data

def load(corpus_name, set_name):
  if corpus_name == 'math':
    return load_math(set_name)
  elif corpus_name == 'shapes':
    return load_shapes(set_name)
  elif corpus_name == "images":
    return load_images(set_name)
  else:
    assert False

if __name__ == '__main__':
  load_math("train.tiny")
