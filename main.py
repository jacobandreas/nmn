#!/usr/bin/env python2

import util

import argparse
from collections import defaultdict
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

# check profiler
if not isinstance(__builtins__, dict) or "profile" not in __builtins__:
    __builtins__.__dict__["profile"] = lambda x: x

def main():
    args = arg_parser.parse_args()

    with open(args.log_config) as log_config_f:
        logging.config.dictConfig(yaml.load(log_config_f))

    with open(args.config) as config_f:
        config = util.Struct(**yaml.load(config_f))

    corpus = importlib.import_module("task.%s" % config.corpus.name)

    train_data = corpus.load_train(config.corpus.train_size)
    val_data = corpus.load_val()

    backend = importlib.import_module("backend.%s" % config.backend)
    model = backend.build_model(config.model, config.opt)

    train_data_grouped = by_query(train_data)
    train_queries = train_data_grouped.keys()
    val_data_grouped = by_query(val_data)
    val_queries = val_data_grouped.keys()

    input_shapes = [d.input.shape for d in (train_data + val_data)]
    max_input_size = tuple(np.max(input_shapes, axis=0))

    for i_iter in range(config.opt.iters):
        do_eval = i_iter % 5 == 0
        np.random.shuffle(train_queries)
        train_loss, train_acc = batched_iter(train_queries, train_data_grouped,
                max_input_size, model, config, train=True, compute_eval=do_eval)
        #np.random.shuffle(train_data)
        #train_loss, train_acc = simple_iter(train_data, model, train=True,
        #        compute_eval=do_eval)
        if do_eval:
            #val_loss, val_acc = simple_iter(val_data, model, compute_eval=True)
            val_loss, val_acc = batched_iter(val_queries, val_data_grouped,
                    max_input_size, model, config, compute_eval=True)
            logging.info("%2.4f  %2.4f  :  %2.4f  %2.4f",
                    train_loss, train_acc, val_loss, val_acc)
        else:
            logging.info("%2.4f", train_loss)


def batched_iter(queries, data_grouped, max_input_size, model, config, 
                 train=False, compute_eval=False):
    batch_loss = 0.
    batch_acc = 0.
    count = 0
    for query in queries:
        query_data = data_grouped[query]
        for batch_start in range(0, len(query_data), config.opt.batch_size):
            batch_data = query_data[batch_start:batch_start+config.opt.batch_size]
            batch_size = len(batch_data)
            count += batch_size
            batch_input = np.zeros((batch_size,) + max_input_size)
            for i, datum in enumerate(batch_data):
                ds = datum.input.shape
                batch_input[i,:ds[0],:ds[1],:ds[2]] = datum.input
            batch_output = np.asarray([datum.output for datum in batch_data])
            
            loss, acc = model.forward(query, batch_input, batch_output,
                                      compute_eval)
            batch_loss += loss * batch_size
            if compute_eval:
                batch_acc += acc
            if train:
                model.train()
            model.clear()
    return batch_loss / count, batch_acc / count

def simple_iter(data, model, train=False, compute_eval=False):
    batch_loss = 0.
    batch_acc = 0.
    for i_datum in range(len(data)):
        query = data[i_datum].query
        batch_data = [data[i_datum]]
        batch_input = np.asarray([d.input for d in batch_data])
        batch_output = np.asarray([d.output for d in batch_data])
        loss, acc = model.forward(query, batch_input, batch_output, compute_eval)
        batch_loss += loss
        if compute_eval:
            batch_acc += acc
        if train:
            model.train()
        model.clear()
    return batch_loss / len(data), batch_acc / len(data)


def by_query(data):
    grouped = defaultdict(list)
    for datum in data:
        grouped[datum.query].append(datum)
    return grouped

if __name__ == "__main__":
    main()
