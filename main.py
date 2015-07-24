#!/usr/bin/env python2

import corpus
from nmn import NMN
from lstm import LSTM
from input_types import *
from output_types import *

from collections import defaultdict
import logging, logging.config
import numpy as np
import theano
import yaml

theano.config.optimizer = "None"

LOG_CONFIG = "log.yaml"
EXPERIMENT_CONFIG = "config.yaml"

if __name__ == "__main__":

  with open(LOG_CONFIG) as log_config_f:
    logging.config.dictConfig(yaml.load(log_config_f))

  with open(EXPERIMENT_CONFIG) as experiment_config_f:
    config = yaml.load(experiment_config_f)["experiment"]

  train_data = corpus.load(config["corpus"], "train.%s" % config["train_size"])
  train_data = train_data[:6000]
  logging.info("loaded train.%s", config["train_size"])
  val_data = corpus.load(config["corpus"], "val")[:1000]
  logging.info("loaded val")

  input_type = eval(config["input_type"])
  output_type = eval(config["output_type"])

  # TODO(jda) cleaner
  model_params = config["model_params"]
  model = globals()[config["model"]](input_type, output_type, model_params)

  train_data_by_query = defaultdict(list)
  for datum in train_data:
    if datum.query[0] == "count": continue
    train_data_by_query[datum.query].append(datum) 

  val_data_by_query = defaultdict(list)
  for datum in val_data:
    if datum.query[0] == "count": continue
    val_data_by_query[datum.query].append(datum)

  train_queries = list(train_data_by_query.keys())

  for i in range(10000):
    np.random.shuffle(train_queries)
    epoch_train_ll = 0.
    epoch_train_acc = 0.
    for query in train_queries:
      data = train_data_by_query[query]
      batch_inputs = np.asarray([datum.input_ for datum in data], dtype=input_type.dtype)
      batch_outputs = np.asarray([datum.output for datum in data], dtype=output_type.dtype)
      train_ll = model.train(query, batch_inputs, batch_outputs)
      batch_pred = model.predict(query, batch_inputs)
      epoch_train_ll += train_ll
      epoch_train_acc += 1. * sum(np.equal(batch_pred, batch_outputs)) / len(data)
    epoch_train_ll /= len(train_queries)
    epoch_train_acc /= len(train_queries)
    
    epoch_val_ll = 0.
    epoch_val_acc = 0.
    for query, data in val_data_by_query.items():
      batch_inputs = np.asarray([datum.input_ for datum in data], dtype=input_type.dtype)
      batch_outputs = np.asarray([datum.output for datum in data], dtype=output_type.dtype)
      val_ll = model.loss(query, batch_inputs, batch_outputs)
      batch_pred = model.predict(query, batch_inputs)
      epoch_val_ll += val_ll
      epoch_val_acc += 1. * sum(np.equal(batch_pred, batch_outputs)) / len(data)


    if val_data_by_query:
        epoch_val_ll /= len(val_data_by_query)
        epoch_val_acc /= len(val_data_by_query)

    logging.info("%0.4f\t%0.4f\t|\t%0.4f\t%0.4f", epoch_train_ll, epoch_val_ll,
            epoch_train_acc, epoch_val_acc)
