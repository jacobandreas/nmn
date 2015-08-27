#!/usr/bin.env python2

import my_layers
from apollocaffe import layers

class IdentityModule:
    def __init__(self, name, input_name, incoming_names, apollo_net):
        self.last_layer_name = "Ident_%s" % name

    def forward(self):
        assert False

class ConvModule:
    def __init__(self, name, hidden_size, input_name, incoming_names, apollo_net):
        self.name = name
        self.hidden_size = hidden_size
        self.input_name = input_name
        assert len(incoming_names) == 0
        self.incoming_names = incoming_names
        self.apollo_net = apollo_net

        self.last_layer_name = "Conv_%s" % name

    def forward(self):
        self.apollo_net.f(layers.Convolution(self.last_layer_name, (1,1), 1,
                                             bottoms=[self.input_name]))

class AnswerModule:
    def __init__(self, name, input_name, incoming_names, apollo_net):
        self.name = name
        self.input_name = input_name
        assert len(incoming_names) == 1
        self.incoming_names = incoming_names
        self.apollo_net = apollo_net


        name_prefix = "Answer_%s__" % name
        self.flatten_layer_name = name_prefix + "flatten"
        self.softmax_layer_name = name_prefix + "softmax"
        self.reshape_layer_name = name_prefix + "reshape"
        self.attention_layer_name = name_prefix + "attention"
        self.ip_layer_name = name_prefix + "ip"
        #self.prediction_layer_name = name_prefix + "pred"

        #self.last_layer_name = self.tile_layer_name
        self.last_layer_name = self.ip_layer_name

    def forward(self):
        input_channels = self.apollo_net.blobs[self.input_name].shape[1]
        incoming_shape = tuple(self.apollo_net.blobs[self.incoming_names[0]].shape)

        self.apollo_net.f(my_layers.Flatten(
                self.flatten_layer_name,
                bottoms=self.incoming_names))

        self.apollo_net.f(layers.Softmax(
                self.softmax_layer_name,
                bottoms=[self.flatten_layer_name]))

        # TODO(jda) is this evil?
        self.apollo_net.blobs[self.softmax_layer_name].reshape(incoming_shape)

        self.apollo_net.f(my_layers.Attention(
                self.attention_layer_name,
                bottoms=[self.softmax_layer_name, self.input_name]))

        self.apollo_net.f(layers.InnerProduct(
                self.ip_layer_name,
                32,
                bottoms=[self.attention_layer_name]))

        #self.apollo_net.f(layers.Softmax(
        #        self.prediction_layer_name,
        #        bottoms=[self.ip_layer_name]))

class DataModule:
    def __init__(self, name, apollo_net):
        self.apollo_net = apollo_net
        self.last_layer_name = name

    def forward(self, data):
        self.apollo_net.f(layers.NumpyData(self.last_layer_name, data=data))

class ClassificationLogLossModule:
    def __init__(self, output_name, apollo_net):
        self.apollo_net = apollo_net
        self.output_name = output_name
        self.target_name = "Target"
        self.loss_name = "Loss"

    def forward(self, target):
        return self.apollo_net.f(layers.SoftmaxWithLoss(
            self.loss_name, bottoms=[self.output_name, self.target_name]))

class ClassificationAccuracyModule:
    def __init__(self, output_name, apollo_net):
        self.apollo_net = apollo_net
        self.output_name = output_name
        self.target_name = "Target"
        self.acc_name = "Accuracy"

    def forward(self, target):
        return self.apollo_net.f(my_layers.Accuracy(
            self.acc_name, bottoms=[self.output_name, self.target_name]))
