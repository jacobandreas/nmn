#!/usr/bin/env python2

from indices import ANSWER_INDEX
import models
import my_layers
from util import Struct

from apollocaffe import ApolloNet, layers
import yaml

class EnsembleModel:
    def __init__(self, config, opt_config):
        self.config = config
        self.opt_config = opt_config

        self.models = []
        for cmodel in config.models:
            with open(cmodel.config) as config_f:
                mconfig = Struct(**yaml.load(config_f))
                model = models.build_model(mconfig.model, mconfig.opt)
            model.load(cmodel.weights)
            self.models.append(model)

        self.n_models = len(self.models)

        self.apollo_net = ApolloNet()

    def forward(self, layout_type, indices, string, input, target, 
                compute_eval=False):
        batch_size = -1

        for i_model, model in enumerate(self.models):
            model.forward(layout_type, indices, string, input, target, compute_eval)
            answer = model.apollo_net.blobs[model.answer_layer].data
            batch_size = answer.shape[0]
            self.apollo_net.f(layers.NumpyData("output_%d" % i_model, answer))

        self.apollo_net.f(layers.Concat(
            "concat", bottoms=["output_%d" % i for i in range(self.n_models)]))

        self.apollo_net.blobs["concat"].reshape(
            (batch_size, self.n_models, len(ANSWER_INDEX), 1))

        self.apollo_net.f(layers.Convolution(
            "merge", (1,1), 1, bottoms=["concat"]))

        self.apollo_net.blobs["merge"].reshape(
            (batch_size, len(ANSWER_INDEX)))

        self.apollo_net.f(layers.NumpyData("target", target))
        loss = self.apollo_net.f(layers.SoftmaxWithLoss("loss", bottoms=["merge", "target"],
                                     normalize=False))
        if compute_eval:
            eval = self.apollo_net.f(my_layers.Accuracy("acc", bottoms=["merge", "target"]))
            return loss, eval
        else: 
            return loss, None

    def train(self):
        self.apollo_net.backward()
        self.apollo_net.update(lr=self.opt_config.learning_rate,
                               momentum=self.opt_config.momentum,
                               clip_gradients=self.opt_config.clip)
        if self.config.train_submodels:
            for model in self.models:
                model.train()

    def clear(self):
        for model in self.models:
            model.clear()
        self.apollo_net.clear_forward()

    def save(self, dest):
        pass
