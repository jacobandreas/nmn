#!/usr/bin/env python2

import corpus
from nmn import NMN
from lstm import LSTM
from input_types import *
from output_types import *
from shapes import *
from images import *
import matplotlib.image
import matplotlib.cm
import scipy.misc

from collections import defaultdict
import logging, logging.config
import numpy as np
import theano
import yaml

#theano.config.optimizer = "None"
#theano.exception_verbosity = "high"

LOG_CONFIG = "log.yaml"
EXPERIMENT_CONFIG = "config.yaml"

if __name__ == "__main__":

  with open(LOG_CONFIG) as log_config_f:
    logging.config.dictConfig(yaml.load(log_config_f))

  with open(EXPERIMENT_CONFIG) as experiment_config_f:
    config = yaml.load(experiment_config_f)["experiment"]

  train_data = corpus.load(config["corpus"], "train.%s" % config["train_size"])
  #train_data = train_data[2:3] #train_data[:10000]
  logging.info("loaded train.%s", config["train_size"])
  val_data = corpus.load(config["corpus"], "val")
  val_data = [] #val_data[:1000]
  logging.info("loaded val")

  input_type = eval(config["input_type"])
  output_type = eval(config["output_type"])

  # TODO(jda) cleaner
  model_params = config["model_params"]
  model = globals()[config["model"]](input_type, output_type, model_params)

  train_data_by_query = defaultdict(list)
  for datum in train_data:
    #if datum.query[0] == "count": continue
    #if datum.query[0] == "shape": continue
    train_data_by_query[datum.query].append(datum) 

  val_data_by_query = defaultdict(list)
  for datum in val_data:
    #if datum.query[0] == "count": continue
    #if datum.query[0] == "shape": continue
    val_data_by_query[datum.query].append(datum)

  train_queries = list(train_data_by_query.keys())

  for i in range(40000):
    np.random.shuffle(train_queries)
    epoch_train_ll = 0.
    epoch_train_acc = 0.
    for query in train_queries:
      query_data = train_data_by_query[query]
      for j in range(len(query_data)):

        data = query_data[j:j+1]
        batch_inputs = np.asarray([datum.input_ for datum in data], dtype=input_type.dtype)
        batch_outputs = np.asarray([datum.output for datum in data], dtype=output_type.dtype)

        diag = model.diagnose(query, batch_inputs)

        #resp = model.response(query, batch_inputs)
        #print resp
        #print model.response(query, batch_inputs)
        #grad = model.grad(query, batch_inputs, batch_outputs)
        #print grad[-1]
        #exit()
        #print resp.shape
        #print resp
        #print np.linalg.norm(resp[:,0])
        #print np.linalg.norm(resp[:,1])

        #train_ll = model.train(query, batch_inputs, batch_outputs, return_norm=False)
        train_ll = model.loss(query, batch_inputs, batch_outputs)

        #logging.debug("norm %0.4f", norm)
        batch_pred = model.predict(query, batch_inputs)
        #print batch_pred, batch_outputs
        epoch_train_ll += train_ll
        #print hash(str(batch_inputs.nonzero()))
        #print query, train_ll, batch_pred
        epoch_train_acc += 1. * sum(np.equal(batch_pred, batch_outputs)) / len(data)

        #diag = scipy.misc.imresize(diag, 10.)
        #matplotlib.image.imsave("diagnose/%s" % data[0].name, diag, cmap=matplotlib.cm.binary)

    epoch_train_ll /= len(train_data) #len(train_queries)
    epoch_train_acc /= len(train_data) #len(train_queries)
    
    epoch_val_ll = 0.
    epoch_val_acc = 0.
    for query, query_data in val_data_by_query.items():
      for j in range(len(query_data)):
        data = query_data[j:j+1]
        batch_inputs = np.asarray([datum.input_ for datum in data], dtype=input_type.dtype)
        batch_outputs = np.asarray([datum.output for datum in data], dtype=output_type.dtype)
        val_ll = model.loss(query, batch_inputs, batch_outputs)
        batch_pred = model.predict(query, batch_inputs)
        epoch_val_ll += val_ll
        #print query, val_ll, batch_pred
        epoch_val_acc += 1. * sum(np.equal(batch_pred, batch_outputs)) / len(data)

    if val_data_by_query:
        epoch_val_ll /= len(val_data)
        epoch_val_acc /= len(val_data)

    logging.info("%0.4f\t%0.4f\t|\t%0.4f\t%0.4f", epoch_train_ll, epoch_val_ll,
            epoch_train_acc, epoch_val_acc)

    if i % 10 == 0:
      pass
      #model.serialize("model.txt")
