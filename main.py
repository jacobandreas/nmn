#!/usr/bin/env python2

import util

import argparse
import importlib
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

    for i_iter in range(config.opt.iters):
        iter_loss = 0
        for i_datum in range(len(train_data)):
            query = train_data[i_datum].query
            batch_data = [train_data[i_datum]]
            batch_input = np.asarray([d.input for d in batch_data])
            batch_output = np.asarray([d.output for d in batch_data])
            iter_loss += model.forward(query, batch_input, batch_output)
            model.update()
        print iter_loss / len(train_data)

if __name__ == "__main__":
    main()
