#!/usr/bin/env python2

# check profiler
if not isinstance(__builtins__, dict) or "profile" not in __builtins__:
    __builtins__.__dict__["profile"] = lambda x: x

from indices import STRING_INDEX, ANSWER_INDEX
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

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-c", "--config", dest="config", required=True,
                        help="model configuration file")
arg_parser.add_argument("-l", "--log-config", dest="log_config", 
                        default="config/log.yml", help="log configuration file")

@profile
def main():
    args = arg_parser.parse_args()

    with open(args.log_config) as log_config_f:
        config_name = args.config.split("/")[-1].split(".")[0]
        log_filename = "%s.log" % config_name
        log_config = yaml.load(log_config_f)
        log_config["handlers"]["fileHandler"]["filename"] = log_filename
        logging.config.dictConfig(log_config)

    with open(args.config) as config_f:
        config = util.Struct(**yaml.load(config_f))

    task = tasks.load_task(config.task)
    model = models.build_model(config.model, config.opt)

    train_layout_types = list(task.train.layout_types)

    for i_iter in range(config.opt.iters):
        do_eval = i_iter % 5 == 0
        np.random.shuffle(train_layout_types)
        train_loss, train_acc = batched_iter(
                task.train, model, config, train=True, compute_eval=do_eval,
                layout_order=train_layout_types)
        if do_eval:
            visualizer.begin(100)
            val_loss, val_acc = batched_iter(
                    task.val, model, config, compute_eval=True)
            visualizer.end()
            test_loss, test_acc = batched_iter(
                    task.test, model, config, compute_eval=True)
            logging.info("%2.4f  %2.4f  %2.4f  :  %2.4f  %2.4f  %2.4f",
                    train_loss, val_loss, test_loss, train_acc, val_acc, test_acc)
        else:
            logging.info("%2.4f", train_loss)



def stack_indices(indices):
    if isinstance(indices[0], tuple):
        head_indices = [i[0] for i in indices]
        tail_indices = [[i[j] for i in indices] for j in range(1, len(indices[0]))]
        tail_indices_rec = [stack_indices(t) for t in tail_indices]
        return (head_indices,) + tuple(tail_indices_rec)
    else:
        return indices


def batched_iter(data, model, config, train=False, compute_eval=False, 
                 layout_order=None):
    batch_loss = 0.
    batch_acc = 0.
    count = 0
    if layout_order is not None:
        layout_types = layout_order
    else:
        layout_types = data.layout_types

    for layout_type in layout_types:
        layout_data = data.by_layout_type[layout_type]
        for batch_start in range(0, len(layout_data), config.opt.batch_size):
            batch_data = layout_data[batch_start:batch_start+config.opt.batch_size]
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
            batch_indices = stack_indices([d.layout.indices for d in batch_data])
            
            loss, acc = model.forward(
                    layout_type, batch_indices, batch_input, batch_output, compute_eval)

            #att_blob = model.apollo_net.blobs["AttAnswer__softmax"].data[0,0,...]
            att_blob = model.apollo_net.blobs["IndexedConv__flatten"].data.reshape((64, 1, 20, 20))
            att_blob = att_blob[0, 0, :first_input.shape[1], :first_input.shape[2]]
            visualizer.show([
                " ".join([STRING_INDEX.get(w) for w in batch_data[0].string]),
                "<img src='../%s' />" % batch_data[0].image_path,
                att_blob,
                #first_input[0,...],
                #first_input.shape[1],
                #first_input.shape[2], 
                ANSWER_INDEX.get(np.argmax(model.apollo_net.blobs[model.answer_layer].data[0,...])),
                ANSWER_INDEX.get(batch_output[0])
            ])

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


if __name__ == "__main__":
    main()
