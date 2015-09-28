import embeddings
import models

from collections import defaultdict
from datum import QueryDatum
import logging
import numpy as np
import os
from query import parse_query
from util import Index

# loading

UNKNOWN = "*unknown*"

def load_train(size):
    return load("train.%s" % size)

def load_val():
    return load("val")

def datum_filter(query, answer):
  if not isinstance(query, tuple) or query[0] not in \
      (
        "count",
        "color",
      ): #, "what"):
    return False
  if isinstance(query[1], tuple):
    return False
  #if query not in (("color", "shirt"), ("color", "cat"), ("color", "train")):
  #  return False
  if " " in answer or "/" in answer or "," in answer:
    return False
  return True

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
  index.index(UNKNOWN)
  return index

def load(set_name):
  short_set_name = set_name.split(".")[0]
  data = []
  answer_index = build_image_answer_index()

  with open("data/images/processed/%s.query" % set_name) as query_f, \
       open("data/images/processed/%s.output" % set_name) as output_f, \
       open("data/images/processed/%s.answer_to_question" % set_name) as a2q_f, \
       open("data/images/processed/%s.question_to_image" % set_name) as q2i_f, \
       open("data/images/processed/%s.image_to_index" % set_name) as i2i_f:

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
    for image_id, index in i2i.items():
      try:
        image = np.load("data/images/Images/%s2014/embedded/%s.npy" %
                (short_set_name, image_names[index]))
        if image.shape[1] <= 5 or image.shape[2] <= 5:
            raise IOError("bad shape")
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
        output = answer_index[UNKNOWN]
      name = image_names[i2i[image_id]]
      #print query, answer.strip(), output,
      #print image_id, name
      datum = QueryDatum(query, inp, output, image_names[i2i[image_id]])
      data.append(datum)
      answer_counter[answer.strip()] += 1

      #break

  logging.info("%d items", len(data))
  logging.info("%d classes", len(answer_index))
  
  count_pairs = sorted([(v, k) for k, v in answer_counter.items()])
  for pair in count_pairs:
      logging.info("%4d: %s" % pair)
  #exit()
  return data

# modules

PREPOSITIONS = set([
    "in", "on", "at", "by"
])

EMBEDDINGS = dict()

def ensure_embeddings(config):
    if len(EMBEDDINGS) > 0:
        return
    embedding_matrix, embedding_index = \
            embeddings.load(config.embedding_path)
    EMBEDDINGS["matrix"] = embedding_matrix
    EMBEDDINGS["index"] = embedding_index

def build_support_module(apollo_net, config):
    return models.modules.NullModule()

def build_module(name, arity, input_name, incoming_names, apollo_net, 
                       config):
    if arity == 1 and name in PREPOSITIONS:
        return models.modules.IdentityModule()
    elif arity == 1 and name == "color":
        return models.modules.AttAnswerModule(name, input_name, incoming_names,
                apollo_net)
    elif arity == 1 and name == "count":
        return models.modules.DenseAnswerModule(name, config.hidden_size,
                incoming_names, apollo_net)
    else:
        assert arity == 0
        ensure_embeddings(config)
        return models.modules.IndexedConvModule(
                name, EMBEDDINGS["matrix"], EMBEDDINGS["index"],
                config.hidden_size, input_name, incoming_names, apollo_net)

def build_loss_module(output_name, apollo_net):
    return models.modules.ClassificationLogLossModule(output_name, apollo_net)

def build_eval_module(output_name, apollo_net):
    return models.modules.ClassificationAccuracyModule(output_name, apollo_net)

def build_reading_module(output_name, apollo_net, config):
    return models.modules.LSTMModule(
            config.hidden_size, output_name, apollo_net)

def get_indices(question, config):
    ensure_embeddings(config)
    indexed = []
    for w in question:
        if w in EMBEDDINGS["index"]:
            index = EMBEDDINGS["index"][w]
        else:
            index = EMBEDDINGS["index"][UNKNOWN]
        indexed.append(w)
    return indexed
