#!/usr/bin/env python2

from indices import STRING_INDEX, ANSWER_INDEX
import my_layers

import apollocaffe
from apollocaffe.layers import *
import numpy as np

def linearize(lin, indices):
    if isinstance(indices, tuple):
        for i in indices[1:]:
            linearize(lin, i)
        lin.append(indices[0])
    else:
        lin.append(indices)

class MonolithicNMNModel:
    def __init__(self, config, opt_config):
        self.config = config
        self.opt_config = opt_config

        self.apollo_net = apollocaffe.ApolloNet()

    def train(self):
        self.apollo_net.backward()
        self.apollo_net.update(
            lr=self.opt_config.learning_rate,
            momentum=self.opt_config.momentum,
            clip_gradients=self.opt_config.clip)

    def clear(self):
        pass

    def forward(self, layout_type, indices, input, target, compute_eval=False):
        net = self.apollo_net
        net.clear_forward()

        question = []
        linearize(question, indices)
        question = np.asarray(question).T

        enc_question = self.f_encode_question(question)
        enc_image    = self.f_encode_image(input)
        
        pred1        = self.f_classify("pred_raw", enc_image, enc_question)
        detection    = self.f_detect("det", enc_image, enc_question)
        pred2        = self.f_classify("pred_det", detection, enc_question)
        attention    = self.f_attend("att", enc_image, detection, enc_question)
        pred3        = self.f_classify("pred_att", attention, enc_question)

        answer       = self.f_gate("gate", [pred1, pred2, pred3], enc_question)
        self.answer_layer = answer

        net.f(NumpyData("target", target))
        loss = net.f(SoftmaxWithLoss("loss", bottoms=[answer, "target"]))
        if compute_eval:
            eval = net.f(my_layers.Accuracy("acc", bottoms=[answer, "target"]))
            return loss, eval
        else:
            return loss, None

    def f_encode_image(self, image):
        net = self.apollo_net
        net.f(NumpyData("image", image))
        return "image"


    def f_encode_question(self, question):
        net = self.apollo_net
        n_batch, n_seq = question.shape

        net.f(NumpyData(
            "lstm_seed", np.zeros((n_batch, self.config.hidden_size))))

        for t in range(n_seq):
            if t == 0:
                prev_hidden = "lstm_seed"
                prev_mem = "lstm_seed"
            else:
                prev_hidden = "lstm_hidden_%d" % (t-1)
                prev_mem = "lstm_mem_%d" % (t-1)
            word    = "word_%d" % t
            wordvec = "wordvec_%d" % t
            concat  = "concat_%d" % t
            lstm    = "lstm_%d" % t
            hidden  = "lstm_hidden_%d" % t
            mem     = "lstm_mem_%d" % t

            net.f(NumpyData("word_%d" % t, question[:, t]))

            net.f(Wordvec(
                wordvec, self.config.hidden_size, len(STRING_INDEX),
                bottoms=[word], param_names=["wordvec"]))

            net.f(Concat(concat, bottoms=[prev_hidden, wordvec]))

            net.f(LstmUnit(
                lstm, bottoms=[concat, prev_mem],
                param_names=["lstm_in_val", "lstm_in_gate", "lstm_forget_gate",
                             "lstm_out_gate"],
                tops=[hidden, mem], num_cells=self.config.hidden_size))

        return hidden

    def f_classify(self, prefix, scene, question):
        net = self.apollo_net

        ip_scene    = prefix + "_ip_scene"
        ip_question = prefix + "_ip_question"
        merge       = prefix + "_ip_merge"
        relu1       = prefix + "_relu1"
        ip2         = prefix + "_ip2"

        net.f(InnerProduct(ip_scene, self.config.hidden_size, bottoms=[scene]))
        net.f(InnerProduct(
            ip_question, self.config.hidden_size, bottoms=[question]))
        net.f(Eltwise(merge, bottoms=[ip_scene, ip_question], operation="SUM"))
        net.f(ReLU(relu1, bottoms=[merge]))
        net.f(InnerProduct(ip2, len(ANSWER_INDEX), bottoms=[relu1]))

        return ip2

    def f_detect(self, prefix, image, question):
        net = self.apollo_net
        channels = net.blobs[image].shape[1]

        ip_weight = prefix + "_ip_weight"
        broadcast = prefix + "_broadcast"
        flatten   = "IndexedConv__flatten" #prefix + "_flatten"

        net.f(InnerProduct(ip_weight, channels, bottoms=[question]))
        net.f(Scalar(broadcast, 0, bottoms=[image, ip_weight]))
        net.f(Convolution(flatten, (1,1), 1, bottoms=[broadcast]))

        return flatten


    def f_attend(self, prefix, image, mask, question):
        net = self.apollo_net
        n_channels = net.blobs[image].shape[1]
        n_batch, n_mask_channels, width, height = net.blobs[mask].shape
        assert n_mask_channels == 1

        reshape1  = prefix + "_reshape1"
        softmax   = prefix + "_softmax"
        reshape2  = prefix + "_reshape2"
        tile      = prefix + "_tile"
        prod      = prefix + "_prod"
        reduction = prefix + "_reduction"

        #net.blobs[mask].reshape((n_batch, width * height))
        net.f(Reshape(reshape1, (n_batch, width*height), bottoms=[mask]))
        net.f(Softmax(softmax, bottoms=[reshape1]))
        #net.blobs[softmax].reshape((n_batch, 1, width, height))
        net.f(Reshape(reshape2, (n_batch, 1, width, height), bottoms=[softmax]))
        net.f(Tile(tile, axis=1, tiles=n_channels, bottoms=[reshape2]))
        net.f(Eltwise(prod, bottoms=[tile, image], operation="PROD"))
        net.f(Reduction(reduction, axis=2, bottoms=[prod]))

        return reduction

    def f_gate(self, prefix, inputs, question):
        net = self.apollo_net
        n_batch = net.blobs[question].shape[0]

        ip        = prefix + "_ip"
        softmax   = prefix + "_softmax"
        concat    = prefix + "_concat"
        reshape1  = prefix + "_reshape1"
        broadcast = prefix + "_broadcast"
        reduction = prefix + "_reduction"
        reshape2  = prefix + "_reshape2"
        
        net.f(InnerProduct(ip, len(inputs), bottoms=[question]))
        net.f(Softmax(softmax, bottoms=[ip]))
        net.f(Concat(concat, bottoms=inputs))
        #net.blobs[concat].reshape((n_batch, len(inputs), len(ANSWER_INDEX), 1))
        net.f(Reshape(
            reshape1, (n_batch, len(inputs), len(ANSWER_INDEX), 1),
            bottoms=[concat]))
        net.f(Scalar(broadcast, 0, bottoms=[reshape1, softmax]))
        net.f(Convolution(reduction, (1,1), 1, bottoms=[broadcast]))
        net.f(Reshape(
            reshape2, (n_batch, len(ANSWER_INDEX)), bottoms=[reduction]))
        #net.blobs[reduction].reshape((n_batch, len(ANSWER_INDEX)))

        return reshape2
