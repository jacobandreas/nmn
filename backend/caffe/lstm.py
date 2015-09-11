#!/usr/bin/env python2

from util import Index

import apollocaffe
from apollocaffe.layers import *
import my_layers
import importlib
import numpy as np

def linearize(query):
    if isinstance(query, tuple):
        return sum([linearize(q) for q in query], [])
    else:
        return [query]

class LSTMModel:
    def __init__(self, config, opt_config):
        self.config = config
        self.opt_config = opt_config

        self.word_index = Index()

        self.apollo_net = apollocaffe.ApolloNet()

    def forward(self, query, input, target, compute_eval=False):
        #print
        question = self.get_question(query)
        indexed = [self.word_index.index(w) for w in question]
        indexed = np.asarray(indexed).reshape((1, -1))

        seq_length = indexed.shape[1]
        batch_size = input.shape[0]

        indexed = np.tile(indexed, (batch_size, 1))

        net = self.apollo_net

        net.clear_forward()


        net.f(NumpyData("lstm_seed", np.zeros((batch_size, 
                                               self.config.hidden_size))))
        padded_input = np.zeros((batch_size, 256, 20, 20))
        padded_input[:,:,:input.shape[2],:input.shape[3]] = input

        net.f(NumpyData("image_data", padded_input))
        #print net.blobs["image_data"].shape

        if self.config.image_before:
            net.f(InnerProduct("ip1_image_before", self.config.hidden_size,
                               bottoms=["image_data"]))
            net.f(ReLU("relu1_image_before", bottoms=["ip1_image_before"]))
            #print net.blobs["relu1_image_before"].shape
            net.f(Concat("concat_image_before", bottoms=["lstm_seed",
                                                         "relu1_image_before"]))
            #print net.blobs["concat_image_before"].shape
            #print net.blobs["lstm_seed"].shape
            net.f(LstmUnit("lstm_image_before", 
                           bottoms=["concat_image_before", "lstm_seed"],
                           param_names=["lstm_input_value", "lstm_input_gate",
                                        "lstm_forget_gate", "lstm_output_gate"],
                           tops=["lstm_hidden_image_before",
                                 "lstm_mem_image_before"],
                           num_cells=self.config.hidden_size))

            #print net.blobs["lstm_hidden_image_before"].shape

        for i in range(seq_length):
            if i == 0:
                if self.config.image_before:
                    prev_hidden = "lstm_hidden_image_before"
                    prev_mem = "lstm_mem_image_before"
                else:
                    prev_hidden = "lstm_seed"
                    prev_mem = "lstm_seed"
            else:
                prev_hidden = "lstm_hidden_%d" % (i-1)
                prev_mem = "lstm_mem_%d" % (i-1)
            word = indexed[:, i]

            net.f(NumpyData("word_%d" % i, word))

            net.f(Wordvec("wordvec_%d" % i, self.config.hidden_size, 2000,
                          bottoms=["word_%d" % i], 
                          param_names=["wordvec_param"]))

            net.f(Concat("concat_%d" % i, bottoms=[prev_hidden, 
                                                   "wordvec_%d" % i]))

            net.f(LstmUnit("lstm_%d" % i, bottoms=["concat_%d" % i, prev_mem],
                           param_names=["lstm_input_value", "lstm_input_gate",
                                        "lstm_forget_gate", "lstm_output_gate"],
                           tops=["lstm_hidden_%d" % i, "lstm_mem_%d" % i],
                           num_cells=self.config.hidden_size))

            #print net.blobs["lstm_hidden_%d" % i].shape

        last_hidden = "lstm_hidden_%d" % (seq_length - 1)
        last_mem = "lstm_mem_%d" % (seq_length - 1)

        if self.config.image_after:
            net.f(InnerProduct("ip1_image_after", self.config.hidden_size,
                               bottoms=["image_data"]))
            net.f(ReLU("relu1_image_after", bottoms=["ip1_image_after"]))
            #print net.blobs[last_hidden].shape
            #print net.blobs["relu1_image_after"].shape
            net.f(Concat("concat_image_after", bottoms=[last_hidden,
                                                        "relu1_image_after"]))
            net.f(LstmUnit("lstm_image_after", 
                           bottoms=["concat_image_after", last_mem],
                           param_names=["lstm_input_value", "lstm_input_gate",
                                        "lstm_forget_gate", "lstm_output_gate"],
                           tops=["lstm_hidden_image_after",
                                 "lstm_mem_image_after"],
                           num_cells=self.config.hidden_size))
            last_hidden = "lstm_hidden_image_after"

        net.f(InnerProduct("ip", self.config.hidden_size, 
                           bottoms=[last_hidden]))

        net.f(ReLU("relu", bottoms=["ip"]))

        net.f(NumpyData("target", target))

        loss = net.f(SoftmaxWithLoss("loss", bottoms=["relu", "target"],
                                     normalize=False))

        if compute_eval:
            eval = net.f(my_layers.Accuracy("acc", bottoms=["relu", "target"]))
            return loss, eval
        else: 
            return loss, None

    def train(self):
        self.apollo_net.backward()
        self.apollo_net.update(lr=self.opt_config.learning_rate,
                               momentum=self.opt_config.momentum,
                               clip_gradients=self.opt_config.clip)

    def clear(self):
        pass

    def get_question(self, query):
        if self.config.question_mode == "raw":
            raise NotImplementedError()
        elif self.config.question_mode == "query":
            return linearize(query)
        else:
            raise NotImplementedError()
