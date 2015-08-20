#!/usr/bin/env python2

import caffe.nmn
from datum import QueryDatum
from query import parse_query
from util import Index

import numpy as np

# loading

def load_train(size):
    #return load("train.%s" % size)
    return load("val")

def load_val():
    return load("val")

def build_image_answer_index():
  index = Index()
  with open("data/images/processed/train.large.output") as output_f:
    for line in output_f:
      index.index(line.strip())
  index.index("UNK")
  return index

def load(set_name):
  data = []
  answer_index = build_image_answer_index()

  with open("data/images/processed/%s.query" % set_name) as query_f, \
       open("data/images/processed/%s.output" % set_name) as output_f, \
       open("data/images/processed/%s.answer_to_question" % set_name) as a2q_f, \
       open("data/images/processed/%s.question_to_image" % set_name) as q2i_f, \
       open("data/images/processed/%s.image_to_index" % set_name) as i2i_f:
    inputs = np.load("data/images/processed/%s.input.npy" % set_name)

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

    images = dict()
    for image_id, index in i2i.items():
      images[image_id] = inputs[index,:,:,:]

    questions = dict()
    for question_id, query in zip(question_ids, query_f):
      questions[question_id] = parse_query(query.strip())

    for answer_id, answer in zip(answer_ids, output_f):
      question_id = a2q[answer_id]
      image_id = q2i[question_id]
      inp = images[image_id]
      query = questions[question_id]

      if not isinstance(query, tuple) or query[0] != "color":
        continue
      if " " in answer or "/" in answer or "," in answer:
        continue

      output = answer_index[answer.strip()]
      if output is None:
        output = answer_index["UNK"]
      datum = QueryDatum(query, inp, output)
      data.append(datum)

  return data

# modules

def build_caffe_module(name, arity):
    return caffe.nmn.Module()
