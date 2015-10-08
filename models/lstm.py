#!/usr/bin/env python2

from indices import STRING_INDEX, ANSWER_INDEX

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

        self.apollo_net = apollocaffe.ApolloNet()

    def forward(self, layout_type, indices, string, input, target, compute_eval=False):
        indexed = string
        seq_length = indexed.shape[1]
        batch_size = input.shape[0]

        image_data = "LSTM__image_data"
        ip1_image_before = "LSTM__ip1_image_before"
        relu1_image_before = "LSTM__relu1_image_before"
        concat_image_before = "LSTM__concat_image_before"
        relu2_image_before = "LSTM__relu2_image_before"
        lstm_image_before = "LSTM__image_before"
        ip1_image_after = "LSTM__ip1_image_after"
        relu1_image_after = "LSTM__relu1_image_after"
        concat_image_after = "LSTM__concat_image_after"
        relu2_image_after = "LSTM__relu2_image_after"
        lstm_image_after = "LSTM__image_after"

        seed_name = "LSTM__seed"
        hidden_name = "LSTM__hidden_%s"
        mem_name = "LSTM__mem_%d"
        word_name = "LSTM__word_%d"
        wordvec_name = "LSTM__wordvec_%s"
        concat_name = "LSTM__concat_%s"
        lstm_name = "LSTM__lstm_%s"
        ip_name = "LSTM__ip"
        relu_name = "LSTM__relu"

        wordvec_param_name = "LSTM__wordvec_param"
        input_value_param_name = "LSTM__input_value_param"
        input_gate_param_name = "LSTM__input_gate_param"
        forget_gate_param_name = "LSTM__forget_gate_param"
        output_gate_param_name = "LSTM__output_gate_param"

        net = self.apollo_net

        net.clear_forward()

        net.f(NumpyData(seed_name, np.zeros((batch_size, 
                                             self.config.hidden_size))))

        net.f(NumpyData(image_data, input))

        if self.config.image_before:
            net.f(InnerProduct(ip1_image_before, self.config.hidden_size,
                               bottoms=[image_data]))
            net.f(ReLU(bottoms=[relu1_image_before]))
            #print net.blobs["relu1_image_before"].shape
            net.f(Concat(concat_image_before, bottoms=[seed_name, 
                                                       relu1_image_before]))
            #print net.blobs["concat_image_before"].shape
            #print net.blobs["seed_name"].shape
            net.f(LstmUnit(lstm_image_before, 
                           bottoms=[concat_image_before, seed_name],
                           param_names=[input_value_param_name, input_gate_param_name,
                                        forget_gate-param_name, output_gate_param_name],
                           tops=[hidden_name % "image_before",
                                 mem_name % "image_before"],
                           num_cells=self.config.hidden_size))

            #print net.blobs["lstm_hidden_image_before"].shape

        for i in range(seq_length):
            if i == 0:
                if self.config.image_before:
                    prev_hidden = hidden_name % "image_before"
                    prev_mem = mem_name % "image_before"
                else:
                    prev_hidden = seed_name
                    prev_mem = seed_name
            else:
                prev_hidden = hidden_name % (i-1)
                prev_mem = mem_name % (i-1)
            word = indexed[:, i]

            net.f(NumpyData(word_name % i, word))

            net.f(Wordvec(wordvec_name % i, self.config.hidden_size, len(STRING_INDEX),
                          bottoms=[word_name % i], 
                          param_names=[wordvec_param_name]))

            net.f(Concat(concat_name % i, bottoms=[prev_hidden, 
                                                   wordvec_name % i]))

            net.f(LstmUnit(lstm_name % i, bottoms=[concat_name % i, prev_mem],
                           param_names=[input_value_param_name, input_gate_param_name,
                                        forget_gate_param_name, output_gate_param_name],
                           tops=[hidden_name % i, mem_name % i],
                           num_cells=self.config.hidden_size))

            #print net.blobs["lstm_hidden_%d" % i].shape

        last_hidden = hidden_name % (seq_length - 1)
        last_mem = mem_name % (seq_length - 1)

        if self.config.image_after:
            net.f(InnerProduct(ip1_image_after, self.config.hidden_size,
                               bottoms=[image_data]))
            net.f(ReLU(relu1_image_after, bottoms=[ip1_image_after]))
            #print net.blobs[last_hidden].shape
            #print net.blobs["relu1_image_after"].shape
            net.f(Concat(concat_image_after, bottoms=[last_hidden,
                                                      relu1_image_after]))
            net.f(LstmUnit(lstm_image_after, 
                           bottoms=[concat_image_after, last_mem],
                           param_names=[input_value_param_name, input_gate_param_name,
                                        forget_gate_param_name, output_gate_param_name],
                           tops=[hidden_name % "image_after",
                                 mem_name % "image_after"],
                           num_cells=self.config.hidden_size))
            last_hidden = hidden_name % "image_after"

        net.f(InnerProduct(ip_name, len(ANSWER_INDEX), bottoms=[last_hidden]))

        net.f(ReLU(relu_name, bottoms=[ip_name]))

        self.answer_layer = relu_name

        net.f(NumpyData("target", target))

        loss = net.f(SoftmaxWithLoss("loss", bottoms=[relu_name, "target"],
                                     normalize=False))

        if compute_eval:
            eval = net.f(my_layers.Accuracy("acc", bottoms=[relu_name, "target"]))
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

    def save(self, dest):
        self.apollo_net.save(dest)

    def load(self, src):
        self.apollo_net.load(src)
