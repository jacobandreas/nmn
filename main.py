#!/usr/bin/env python2

# check profiler
if not isinstance(__builtins__, dict) or "profile" not in __builtins__:
    __builtins__.__dict__["profile"] = lambda x: x

from indices import STRING_INDEX, ANSWER_INDEX, NULL
import models
import tasks
import util
from visualizer import visualizer

import argparse
from collections import defaultdict
import importlib
import itertools
import logging.config
import numpy as np
import yaml

KBEST=5

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-c", "--config", dest="config", required=True,
                        help="model configuration file")
arg_parser.add_argument("-l", "--log-config", dest="log_config", 
                        default="config/log.yml", help="log configuration file")

@profile
def main():
    args = arg_parser.parse_args()
    config_name = args.config.split("/")[-1].split(".")[0]

    with open(args.log_config) as log_config_f:
        log_filename = "logs/%s.log" % config_name
        log_config = yaml.load(log_config_f)
        log_config["handlers"]["fileHandler"]["filename"] = log_filename
        logging.config.dictConfig(log_config)

    with open(args.config) as config_f:
        config = util.Struct(**yaml.load(config_f))

    task = tasks.load_task(config.task)
    model = models.build_model(config.model, config.opt)

    for i_iter in range(config.opt.iters):
        do_eval = i_iter % 5 == 0
        train_loss, train_acc = batched_iter(
                task.train, model, config, train=True, compute_eval=do_eval)
        if do_eval:
            val_loss, val_acc = batched_iter(
                    task.val, model, config, compute_eval=True)
            visualizer.begin(config_name, 100)
            test_loss, test_acc = batched_iter(
                    task.test, model, config, compute_eval=True)
            visualizer.end()
            logging.info("%5d  :  %2.4f  %2.4f  %2.4f  :  %2.4f  %2.4f  %2.4f",
                    i_iter, train_loss, val_loss, test_loss, train_acc, val_acc, 
                    test_acc)
            model.save("saves/%s_%d.caffemodel" % (config_name, i_iter))
        else:
            logging.info("%5d  :  %2.4f", i_iter, train_loss)
        exit()



def stack_indices(indices):
    if isinstance(indices[0], tuple):
        head_indices = [i[0] for i in indices]
        tail_indices = [[i[j] for i in indices] for j in range(1, len(indices[0]))]
        tail_indices_rec = [stack_indices(t) for t in tail_indices]
        return (head_indices,) + tuple(tail_indices_rec)
    else:
        return indices


def batched_iter(data, model, config, train=False, compute_eval=False):
    batch_loss = 0.
    batch_acc = 0.
    count = 0

    if config.opt.batch_by == "layout":
        grouped_data = data.by_layout_type
    elif config.opt.batch_by == "length":
        grouped_data = data.by_string_length
    elif config.opt.batch_by == "both":
        grouped_data = data.by_layout_and_length
    keys = list(grouped_data.keys())
    np.random.shuffle(keys)

    for key in keys:
        key_data = list(grouped_data[key])
        np.random.shuffle(key_data)
        for batch_start in range(0, len(key_data), config.opt.batch_size):
            batch_data = key_data[batch_start:batch_start+config.opt.batch_size]
            batch_size = len(batch_data)
            # TODO FIX
            if batch_size < config.opt.batch_size:
                continue
            #count += batch_size
            count += 1

            first_input = batch_data[0].load_input()
            batch_input = np.zeros((
                    len(batch_data), first_input.shape[0],
                    config.task.pad_to_width, config.task.pad_to_height))
            for i, datum in enumerate(batch_data):
                datum_input = datum.load_input()
                channels, width, height = datum_input.shape
                batch_input[i,:,:width,:height] = datum_input
            batch_output = np.asarray([d.outputs[0] for d in batch_data])

            batch_indices = None
            layout_type = None
            strings = None
            if config.opt.batch_by in ("layout", "both"):
                batch_indices = stack_indices([d.layout.indices for d in batch_data])
                layout_type = key if config.opt.batch_by == "layout" else key[0]

            max_string_len = max(len(d.string) for d in batch_data)
            min_string_len = min(len(d.string) for d in batch_data)
            strings = [[STRING_INDEX[NULL]] * (max_string_len - len(d.string)) + d.string for d in batch_data]
            strings = np.asarray(strings)
            
            loss, acc = model.forward(
                    layout_type, batch_indices, strings, batch_input, 
                    batch_output, compute_eval)

            #with open("preds_%s.txt" % train, "a") as rerank_f:
            #    for i in range(len(batch_datum))
            #        print >>rerank_f, " ".join([STRING_INDEX.get(w) for w in batch_data[i].string[1:-1]]),
            #        print >>rerank_f, ANSWER_INDEX.get(datum.outputs[i]),

            #        preds = np.argsort(-model.apollo_net.blobs[model.answer_layer].data[i,...])
            #        pweights = -np.sort(-model.apollo_net.blobs[model.answer_layer].data[i,...])
            #        for k in range(KBEST):
            #            print >>rerank_f, ANSWER_INDEX.get(preds[k]), pweights[k],

            vis(model, batch_data, batch_output, first_input)

            batch_loss += loss
            if compute_eval:
                batch_acc += acc
            if train:
                model.train()
            model.clear()
    if count == 0:
        return 0, 0
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

def vis(model, batch_data, batch_output, first_input):
    fields = []
    fields.append(batch_data[0].raw_query)
    fields.append(" ".join([STRING_INDEX.get(w) for w in batch_data[0].string[1:-1]]))
    fields.append("<img src='../../%s' />" % batch_data[0].image_path)

    if hasattr(model, "apollo_net") and hasattr(model, "attention_layer"):
        att_data = model.apollo_net \
                        .blobs[model.attention_layer] \
                        .data[0] \
                        .reshape((20, 20))[:first_input.shape[1], 
                                           :first_input.shape[2]]
        fields.append(att_data)

    if hasattr(model, "apollo_net") and hasattr(model, "answer_layer"):
        preds = np.argsort(-model.apollo_net.blobs[model.answer_layer].data[0,...])
        predstrs = ", ".join([ANSWER_INDEX.get(p) for p in preds[:KBEST]]),
        fields.append(predstrs)

    fields.append(ANSWER_INDEX.get(batch_output[0]))
    visualizer.show(fields)

if __name__ == "__main__":
    main()
