#!/usr/bin/env python2

import util

import argparse
import importlib
import itertools
import logging.config
import numpy as np
import yaml

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-c", "--config", dest="config", required=True,
                        help="model configuration file")
arg_parser.add_argument("-l", "--log-config", dest="log_config", 
                        default="config/log.yml", help="log configuration file")

def main():
    args = arg_parser.parse_args()

    with open(args.log_config) as log_config_f:
        logging.config.dictConfig(yaml.load(log_config_f))

    with open(args.config) as config_f:
        config = util.Struct(**yaml.load(config_f))

    corpus = importlib.import_module("task.%s" % config.corpus.name)

    train_data = corpus.load_train(config.corpus.train_size)
    val_data = corpus.load_val()

    backend = importlib.import_module(config.backend)
    model = backend.build_model(config.model, config.opt)

    train_data_by_query = itertools.

    for i_iter in range(config.opt.iters):
        train_loss = 0.
        train_acc = 0.
        for i_datum in range(len(train_data)):
            query = train_data[i_datum].query
            batch_data = [train_data[i_datum]]
            batch_input = np.asarray([d.input for d in batch_data])
            batch_output = np.asarray([d.output for d in batch_data])
            loss, acc = model.forward(query, batch_input, batch_output)
            train_loss += loss
            train_acc += acc
            model.train()
            model.clear()
        train_loss /= len(train_data)
        train_acc /= len(train_data)

        val_loss = 0.
        val_acc = 0.
        for i_datum in range(len(val_data)):
            query = val_data[i_datum].query
            batch_data = [val_data[i_datum]]
            batch_input = np.asarray([d.input for d in batch_data])
            batch_output = np.asarray([d.output for d in batch_data])
            loss, acc = model.forward(query, batch_input, batch_output)
            val_loss += loss
            val_acc += acc
            model.clear()
        val_loss /= len(val_data)
        val_acc /= len(val_data)

        print "%2.4f  %2.4f  |  %2.4f  %2.4f" % (train_loss, train_acc,
                                                 val_loss, val_acc)


if __name__ == "__main__":
    main()
