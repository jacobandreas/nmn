#!/usr/bin/env python2

import corpus
from nmn import NMN

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
  logging.info("loaded train.%s", config["train_size"])
  val_data = corpus.load(config["corpus"], "val")[:1000]
  logging.info("loaded val")

  # TODO(jda) cleaner
  model_params = config["model_params"]
  model = globals()[config["model"]](model_params)

  train_data_by_query = defaultdict(list)
  for datum in train_data:
    train_data_by_query[datum.query].append(datum) 

  val_data_by_query = defaultdict(list)
  for datum in val_data:
    val_data_by_query[datum.query].append(datum)

  train_queries = train_data_by_query.keys()
  for i in range(10000):
    np.random.shuffle(train_queries)
    epoch_train_loss = 0.
    for query in train_queries:
      data = train_data_by_query[query]
      batch_inputs = np.asarray([datum.input_ for datum in data], dtype=theano.config.floatX)
      batch_outputs = np.asarray([datum.output for datum in data], dtype=theano.config.floatX)
      train_loss = model.train(query, batch_inputs, batch_outputs)
      epoch_train_loss += train_loss
    epoch_train_loss /= len(train_queries)
    
    epoch_val_loss = 0.
    for query, data in val_data_by_query.items():
      batch_inputs = np.asarray([datum.input_ for datum in data], dtype=theano.config.floatX)
      batch_outputs = np.asarray([datum.output for datum in data], dtype=theano.config.floatX)
      val_loss = model.loss(query, batch_inputs, batch_outputs)
      epoch_val_loss += val_loss
    epoch_val_loss /= len(val_data_by_query)

    logging.info("%0.4f\t%0.4f", epoch_train_loss, epoch_val_loss)
