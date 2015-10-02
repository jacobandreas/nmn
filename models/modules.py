#!/usr/bin.env python2

from indices import LAYOUT_INDEX, ANSWER_INDEX
import my_layers
from util import Index

from apollocaffe import layers
import numpy as np

class NullModule:
    def forward(self):
        pass

class IdentityModule:
    def __init__(self, name, input_name, incoming_names, apollo_net):
        self.output_name = "Ident_%s" % name

    def forward(self):
        assert False

class LSTMModule:
    # index is shared across module instances
    index = Index()

    def __init__(self, hidden_size, incoming_name, apollo_net):
        self.hidden_size = hidden_size
        self.incoming_name = incoming_name
        self.apollo_net = apollo_net

        self.seed_name = "LSTM__seed"
        self.hidden_name = "LSTM__hidden_%d"
        self.mem_name = "LSTM__mem_%d"
        self.word_name = "LSTM__word_%d"
        self.wordvec_name = "LSTM__wordvec_%d"
        self.concat_name = "LSTM__concat_%d"
        self.lstm_name = "LSTM__lstm_%d"
        self.ip_name = "LSTM__ip"
        self.relu_name = "LSTM__relu"
        self.sum_name = "LSTM__sum"

        self.output_name = self.sum_name

        self.wordvec_param_name = "LSTM__wordvec_param"
        self.input_value_param_name = "LSTM__input_value_param"
        self.input_gate_param_name = "LSTM__input_gate_param"
        self.forget_gate_param_name = "LSTM__forget_gate_param"
        self.output_gate_param_name = "LSTM__output_gate_param"

    def forward(self, tokens):
        tokens = np.asarray(tokens).T
        net = self.apollo_net

        net.f(layers.NumpyData(
            self.seed_name, np.zeros((tokens.shape[0], self.hidden_size))))

        for t in range(tokens.shape[1]):
            word_name = self.word_name % t
            wordvec_name = self.wordvec_name % t
            concat_name = self.concat_name % t
            lstm_name = self.lstm_name % t
            hidden_name = self.hidden_name % t
            mem_name = self.mem_name % t
            if t == 0:
                prev_hidden = self.seed_name
                prev_mem = self.seed_name
            else:
                prev_hidden = self.hidden_name % (t - 1)
                prev_mem = self.mem_name % (t - 1)

            net.f(layers.NumpyData(word_name, np.asarray(tokens[:,t])))
            net.f(layers.Wordvec(
                wordvec_name, self.hidden_size, len(LAYOUT_INDEX),
                bottoms=[word_name], param_names=[self.wordvec_param_name]))

            net.f(layers.Concat(concat_name, bottoms=[prev_hidden, wordvec_name]))
            net.f(layers.LstmUnit(
                lstm_name, bottoms=[concat_name, prev_mem],
                param_names=[self.input_value_param_name,
                             self.input_gate_param_name,
                             self.forget_gate_param_name,
                             self.output_gate_param_name],
                tops=[hidden_name, mem_name], num_cells=self.hidden_size))

        net.f(layers.InnerProduct(
            self.ip_name, len(ANSWER_INDEX), bottoms=[hidden_name]))
        net.f(layers.ReLU(self.relu_name, bottoms=[self.ip_name]))
        net.f(layers.Eltwise(
            self.sum_name, bottoms=[self.relu_name, self.incoming_name],
            operation="SUM"))

class IndexedConvModule:
    def __init__(self, hidden_size, input_name, apollo_net):
        self.hidden_size = hidden_size
        self.input_name = input_name
        self.apollo_net = apollo_net

        self.dropout_name = "IndexedConv__dropout"
        self.indices_name = "IndexedConv__indices"
        self.vector_name = "IndexedConv__vec"
        self.conv1_name = "IndexedConv__conv1"
        #self.relu1_name = "IndexedConv__relu1"
        self.scalar_name = "IndexedConv__scalar"
        self.flatten_name = "IndexedConv__flatten"
        self.conv2_name = "IndexedConv__conv2" 
        self.relu2_name = "IndexedConv__relu2"
        #self.output_name = self.relu2_name
        self.output_name = self.flatten_name

    @profile
    def forward(self, indices):
        batch_size, channels, width, height = self.apollo_net.blobs[self.input_name].shape

        #self.apollo_net.f(layers.NumpyData(self.indices_name, indices))

        #self.apollo_net.f(layers.Wordvec(
        #    self.vector_name, self.hidden_size, len(LAYOUT_INDEX),
        #    bottoms=[self.indices_name]))

        #self.apollo_net.f(layers.Convolution(
        #    self.conv1_name, (1,1), self.hidden_size, bottoms=[self.input_name]))

        #t_width_name = "t_width_%d" % width
        #t_height_name = "t_height_%d" % height

        #self.apollo_net.blobs[self.vector_name].reshape((batch_size, self.hidden_size, 1, 1))
        #self.apollo_net.f(layers.Tile(
        #    t_width_name, bottoms=[self.vector_name], axis=2, tiles=width))
        #self.apollo_net.f(layers.Tile(
        #    t_height_name, bottoms=[t_width_name], axis=3, tiles=height))

        #self.apollo_net.f(layers.Eltwise(
        #    self.bc_sum_name, bottoms=[t_height_name, self.conv1_name],
        #    operation="SUM"))

        #self.apollo_net.f(layers.ReLU(
        #    self.relu1_name, bottoms=[self.bc_sum_name]))

        #self.apollo_net.f(layers.Convolution(
        #    self.conv2_name, (1,1), 1, bottoms=[self.relu1_name]))
        #    
        #self.apollo_net.f(layers.ReLU(
        #    self.relu2_name, bottoms=[self.conv2_name]))

        self.apollo_net.f(layers.NumpyData(self.indices_name, indices))

        self.apollo_net.f(layers.Wordvec(
            self.vector_name, channels, len(LAYOUT_INDEX),
            bottoms=[self.indices_name]))

        #t_width_name = "t_width_%d" % width
        #t_height_name = "t_height_%d" % height

        #self.apollo_net.blobs[self.vector_name].reshape((batch_size, channels, 1, 1))
        #self.apollo_net.f(layers.Tile(
        #    t_width_name, bottoms=[self.vector_name], axis=2, tiles=width))
        #self.apollo_net.f(layers.Tile(
        #    t_height_name, bottoms=[t_width_name], axis=3, tiles=height))

        #self.apollo_net.f(layers.Dropout(self.dropout_name, 0.5,
        #    bottoms=[self.input_name]))

        #self.apollo_net.f(my_layers.FeatureDot(
        #    self.bc_sum_name, bottoms=[self.vector_name, self.input_name]))
        
        self.apollo_net.f(layers.Scalar(self.scalar_name, 0,
            bottoms=[self.input_name, self.vector_name]))

        self.apollo_net.f(layers.Convolution(self.flatten_name, (1,1), 1,
            bottoms=[self.scalar_name]))

class DenseAnswerModule:
    def __init__(self, hidden_size, incoming_names, apollo_net):
        self.hidden_size = hidden_size
        assert len(incoming_names) == 1
        self.incoming_name = incoming_names[0]
        self.apollo_net = apollo_net

        name_prefix = "DenseAnswer__"
        self.conv1_name = name_prefix + "conv1"
        self.relu1_name = name_prefix + "relu1"
        #self.conv2_name = name_prefix + "conv2"
        #self.relu2_name = name_prefix + "relu2"
        self.collapse_name = name_prefix + "collapse"
        self.ip_name = name_prefix + "ip"

        self.output_name = self.ip_name

    @profile
    def forward(self, indices):
        #self.apollo_net.f(layers.Convolution(
        #    self.conv1_name, (5, 5), 1, bottoms=[self.incoming_name]))
        #self.apollo_net.f(layers.ReLU(self.relu1_name, bottoms=[self.conv1_name]))
        #self.apollo_net.f(my_layers.Collapse(
        #    self.collapse_name, bottoms=[self.relu1_name]))
        self.apollo_net.f(layers.InnerProduct(
            self.ip_name, len(ANSWER_INDEX), bottoms=[self.incoming_name]))

class AttAnswerModule:
    def __init__(self, hidden_size, input_name, incoming_names, apollo_net):
        self.input_name = input_name
        self.hidden_size = hidden_size
        assert len(incoming_names) == 1
        self.incoming_names = incoming_names
        self.apollo_net = apollo_net

        name_prefix = "AttAnswer__"
        self.flatten_layer_name = name_prefix + "flatten"
        self.softmax_layer_name = name_prefix + "softmax"
        self.reshape_layer_name = name_prefix + "reshape"
        self.attention_layer_name = name_prefix + "attention"
        self.indices_name = name_prefix + "indices"
        self.bias_name = name_prefix + "bias"
        self.ip_layer_name = name_prefix + "ip"
        self.sum_name = name_prefix + "sum"

        self.tile_name = name_prefix + "tile"
        self.reduction_name = name_prefix + "reduction"
        #self.prediction_layer_name = name_prefix + "pred"

        #self.output_name = self.tile_layer_name
        self.output_name = self.sum_name

    @profile
    def forward(self, indices):
        input_channels = self.apollo_net.blobs[self.input_name].shape[1]
        batch_size, mask_channels, width, height = self.apollo_net.blobs[self.incoming_names[0]].shape
        assert mask_channels == 1
        flat_shape = (batch_size, width * height)

        # TODO(jda) is this evil?
        self.apollo_net.blobs[self.incoming_names[0]].reshape(flat_shape)

        self.apollo_net.f(layers.Softmax(
                self.softmax_layer_name,
                bottoms=[self.incoming_names[0]]))

        #self.apollo_net.blobs[self.softmax_layer_name].reshape(
        #        (batch_size, channels, width, height))
        self.apollo_net.blobs[self.softmax_layer_name].reshape(
                (batch_size, 1, width, height))

        self.apollo_net.f(layers.Tile(
                self.tile_name, axis=1, tiles=input_channels,
                bottoms=[self.softmax_layer_name]))

        self.apollo_net.f(layers.Eltwise(
            self.attention_layer_name, bottoms=[self.tile_name, self.input_name],
            operation="PROD"))

        self.apollo_net.f(layers.Reduction(
            self.reduction_name, axis=2, bottoms=[self.attention_layer_name]))
        #self.apollo_net.blobs[self.attention_layer_name].reshape((batch_size,
        #    input_channels, width * height))
        #self.apollo_net.f(layers.InnerProduct(
        #    self.reduction_name, 1, 
        #    axis=2, 
        #    bottoms=[self.attention_layer_name]))
        #    #weight_filler=layers.Filler("constant", 1),
        #    #bias_filler=layers.Filler("constant", 0)))
        #self.apollo_net.blobs[self.reduction_name].reshape((batch_size, input_channels))

        #print self.apollo_net.blobs[self.attention_layer_name].shape
        #print self.apollo_net.blobs[self.reduction_name].shape
        #exit()

        #self.apollo_net.f(my_layers.Attention(
        #        self.attention_layer_name,
        #        bottoms=[self.softmax_layer_name, self.input_name]))


        self.apollo_net.f(layers.NumpyData(self.indices_name, indices))

        self.apollo_net.f(layers.Wordvec(
                self.bias_name, len(ANSWER_INDEX), len(LAYOUT_INDEX),
                bottoms=[self.indices_name]))

        self.apollo_net.f(layers.InnerProduct(
                self.ip_layer_name,
                len(ANSWER_INDEX),
                bottoms=[self.reduction_name]))

        #self.apollo_net.f(layers.ReLU(self.ip_layer_name + "_relu",
        #    bottoms=[self.ip_layer_name]))

        #self.apollo_net.f(layers.InnerProduct(
        #        self.ip_layer_name + "_ip2",
        #        len(ANSWER_INDEX),
        #        bottoms=[self.ip_layer_name + "_relu"]))

        self.apollo_net.f(layers.Eltwise(
                self.sum_name, bottoms=[self.bias_name, self.ip_layer_name],
                operation="SUM"))

class DataModule:
    def __init__(self, name, apollo_net, proj_size=None, dropout=False):
        self.apollo_net = apollo_net
        self.output_name = name
        self.proj_size = proj_size
        self.dropout = dropout

    @profile
    def forward(self, data):
        if self.dropout:
            self.apollo_net.f(layers.NumpyData(self.output_name + "_pre", data=data))
            self.apollo_net.f(layers.Dropout(self.output_name, 0.15,
                bottoms=[self.output_name + "_pre"]))
        else:
            self.apollo_net.f(layers.NumpyData(self.output_name, data=data))

class ClassificationLogLossModule:
    def __init__(self, output_name, apollo_net):
        self.apollo_net = apollo_net
        self.output_name = output_name
        self.target_name = "Target"
        self.loss_name = "Loss__" + output_name

    @profile
    def forward(self, target):
        loss = self.apollo_net.f(layers.SoftmaxWithLoss(
            self.loss_name, bottoms=[self.output_name, self.target_name]))
        return loss

class ClassificationAccuracyModule:
    def __init__(self, output_name, apollo_net):
        self.apollo_net = apollo_net
        self.output_name = output_name
        self.target_name = "Target"
        self.acc_name = "Accuracy"

    @profile
    def forward(self, target):
        return self.apollo_net.f(my_layers.Accuracy(
            self.acc_name, bottoms=[self.output_name, self.target_name]))
