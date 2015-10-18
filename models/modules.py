#!/usr/bin.env python2

from indices import LAYOUT_INDEX, STRING_INDEX, ANSWER_INDEX
import my_layers
from util import Index

from apollocaffe import layers
import numpy as np

class NullModule:
    def forward(self):
        pass

class BOWModule:
    def __init__(self, incoming_name, apollo_net):
        self.incoming_name = incoming_name
        self.apollo_net = apollo_net
        
        self.word_name = "BOW__word_%d"
        self.wordvec_name = "BOW__wordvec_%d"
        self.sum_name = "BOW__sum"

        self.wordvec_param_name = "BOW__wordvec_param"

        self.output_name = self.sum_name

    def forward(self, tokens):
        net = self.apollo_net

        for t in range(tokens.shape[1]):
            word_name = self.word_name % t
            wordvec_name = self.wordvec_name % t
            
            net.f(layers.NumpyData(word_name, np.asarray(tokens[:,t])))
            net.f(layers.Wordvec(
                wordvec_name, len(ANSWER_INDEX), len(STRING_INDEX),
                bottoms=[word_name], param_names=[self.wordvec_param_name]))

        word_bottoms = [self.wordvec_name % t for t in range(tokens.shape[1])]
        bottoms = word_bottoms + [self.incoming_name]

        net.f(layers.Eltwise(self.sum_name, bottoms=bottoms, operation="SUM"))

class LSTMModule:
    def __init__(self, hidden_size, incoming_name, keep_training, apollo_net):
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

        #self.output_name = self.relu_name
        self.output_name = self.sum_name

        self.wordvec_param_name = "LSTM__wordvec_param"
        self.input_value_param_name = "LSTM__input_value_param"
        self.input_gate_param_name = "LSTM__input_gate_param"
        self.forget_gate_param_name = "LSTM__forget_gate_param"
        self.output_gate_param_name = "LSTM__output_gate_param"

        if keep_training:
            self.param_mult = 1.0
        else:
            self.param_mult = 0.0

    @profile
    def forward(self, tokens):
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
                wordvec_name, self.hidden_size, len(STRING_INDEX),
                bottoms=[word_name], param_names=[self.wordvec_param_name],
                param_lr_mults=[self.param_mult]))

            net.f(layers.Concat(concat_name, bottoms=[prev_hidden, wordvec_name]))
            net.f(layers.LstmUnit(
                lstm_name, bottoms=[concat_name, prev_mem],
                param_names=[self.input_value_param_name,
                             self.input_gate_param_name,
                             self.forget_gate_param_name,
                             self.output_gate_param_name],
                param_lr_mults=[self.param_mult] * 4,
                tops=[hidden_name, mem_name], num_cells=self.hidden_size))

        net.f(layers.InnerProduct(
            self.ip_name, len(ANSWER_INDEX), bottoms=[hidden_name],
            param_lr_mults=[self.param_mult] * 2))
        net.f(layers.ReLU(self.relu_name, bottoms=[self.ip_name]))
        net.f(layers.Eltwise(
            self.sum_name, bottoms=[self.relu_name, self.incoming_name],
            operation="SUM"))

class DetectModule:
    def __init__(self, position, hidden_size, input_name, apollo_net):
        self.hidden_size = hidden_size
        self.input_name = input_name
        self.apollo_net = apollo_net

        name_prefix = "Detect_%d__" % position
        self.hidden_name = name_prefix + "hidden"
        self.indices_name = name_prefix + "indices"
        self.vector_name = name_prefix + "vec"
        self.proj_vector_name = name_prefix + "proj_vec"
        self.proj_vector_relu_name = name_prefix + "proj_vec_relu"
        self.scalar_name = name_prefix + "scalar"
        self.flatten_name = name_prefix + "flatten"

        self.shared_name = name_prefix + "shared"
        self.sum_name = name_prefix + "sum"

        #self.output_name = self.sum_name
        self.output_name = self.flatten_name
        #self.output_name = self.shared_name

    @profile
    def forward(self, indices):
        batch_size, channels, width, height = self.apollo_net.blobs[self.input_name].shape

        #self.apollo_net.f(layers.Convolution(self.hidden_name, (1,1),
        #    self.hidden_size, bottoms=[self.input_name]))

        self.apollo_net.f(layers.NumpyData(self.indices_name, indices))

        self.apollo_net.f(layers.Wordvec(
            self.vector_name, 
            #self.hidden_size, 
            channels,
            #64,
            len(LAYOUT_INDEX),
            bottoms=[self.indices_name]))
            #weight_filler=layers.Filler("uniform", 0.001)))

        #self.apollo_net.f(layers.InnerProduct(
        #    self.proj_vector_name, channels, bottoms=[self.vector_name]))

        #self.apollo_net.f(layers.ReLU(
        #    self.proj_vector_relu_name, bottoms=[self.proj_vector_name]))

        self.apollo_net.f(layers.Scalar(self.scalar_name, 0,
            #bottoms=[self.hidden_name, self.vector_name]))
            bottoms=[self.input_name, self.vector_name]))
            #bottoms=[self.input_name, self.proj_vector_relu_name]))

        self.apollo_net.f(layers.Convolution(self.flatten_name, (1,1), 1,
            bottoms=[self.scalar_name]))
            #weight_filler=layers.Filler("constant", 1), 
            #bias_filler=layers.Filler("constant", 0.0),
            #param_lr_mults=[0.0, 0.0]))

        #### self.apollo_net.f(layers.Convolution(self.shared_name, (1,1), 1,
        ####     bottoms=[self.input_name],
        ####     weight_filler=layers.Filler("uniform", 0.001)))

        ## self.apollo_net.f(layers.Eltwise(self.sum_name, operation="SUM",
        ##     bottoms=[self.flatten_name, self.shared_name]))
        

class ConjModule:
    def __init__(self, position, incoming_names, apollo_net):
        self.incoming_names = incoming_names
        self.apollo_net = apollo_net

        name_prefix = "Conj_%d__" % position
        self.min_name = name_prefix + "min"

        self.output_name = self.min_name

    @profile
    def forward(self, indices):
        self.apollo_net.f(layers.Eltwise(
            self.min_name, operation="SUM", bottoms=self.incoming_names))

class RedetectModule:
    def __init__(self, position, incoming_name, apollo_net):
        self.incoming_name = incoming_name
        self.apollo_net = apollo_net

        self.output_name = incoming_name

    @profile
    def forward(self, indices):
        pass


class DenseAnswerModule:
    def __init__(self, position, hidden_size, incoming_names, apollo_net):
        self.hidden_size = hidden_size
        assert len(incoming_names) == 1
        self.incoming_name = incoming_names[0]
        self.apollo_net = apollo_net

        name_prefix = "DenseAnswer_%d__" % position
        self.ip_name = name_prefix + "ip"

        self.output_name = self.ip_name

    @profile
    def forward(self, indices):
        self.apollo_net.f(layers.InnerProduct(
            self.ip_name, len(ANSWER_INDEX), bottoms=[self.incoming_name]))

class AttAnswerModule:
    def __init__(self, position, hidden_size, input_name, incoming_names, apollo_net):
        self.input_name = input_name
        self.hidden_size = hidden_size
        assert len(incoming_names) == 1
        self.incoming_names = incoming_names
        self.apollo_net = apollo_net

        name_prefix = "AttAnswer_%d__" % position
        self.hidden_name = name_prefix + "hidden"
        self.softmax_name = name_prefix + "softmax"
        self.tile_name = name_prefix + "tile"
        self.attention_name = name_prefix + "attention"
        self.reduction_name = name_prefix + "reduction"
        self.indices_name = name_prefix + "indices"
        self.bias_name = name_prefix + "bias"
        self.ip_name = name_prefix + "ip"
        self.sum_name = name_prefix + "sum"

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
                self.softmax_name,
                bottoms=[self.incoming_names[0]]))

        self.apollo_net.blobs[self.softmax_name].reshape(
                (batch_size, 1, width, height))

        self.apollo_net.f(layers.Tile(
                #self.tile_name, axis=1, tiles=self.hidden_size,
                self.tile_name, axis=1, tiles=input_channels,
                bottoms=[self.softmax_name]))

        #self.apollo_net.f(layers.Convolution(
        #    self.hidden_name, (1, 1), self.hidden_size, bottoms=[self.input_name]))

        self.apollo_net.f(layers.Eltwise(
            self.attention_name, bottoms=[self.tile_name, self.input_name],
            operation="PROD"))

        self.apollo_net.f(layers.Reduction(
            self.reduction_name, axis=2, bottoms=[self.attention_name]))

        self.apollo_net.f(layers.NumpyData(self.indices_name, indices))

        self.apollo_net.f(layers.Wordvec(
                self.bias_name, len(ANSWER_INDEX), len(LAYOUT_INDEX),
                bottoms=[self.indices_name]))

        self.apollo_net.f(layers.InnerProduct(
                self.ip_name,
                len(ANSWER_INDEX),
                bottoms=[self.reduction_name]))

        self.apollo_net.f(layers.Eltwise(
                self.sum_name, bottoms=[self.bias_name, self.ip_name],
                operation="SUM"))

class DataModule:
    def __init__(self, name, apollo_net):
        self.output_name = name
        self.apollo_net = apollo_net

    @profile
    def forward(self, data):
        if self.output_name not in self.apollo_net.blobs.keys():
            self.apollo_net.f(layers.NumpyData(self.output_name, data=data))
        else:
            self.apollo_net.blobs[self.output_name].data[:] = data

class ImageDataModule:
    def __init__(self, name, apollo_net, proj_size=None):
        self.apollo_net = apollo_net
        self.proj_size = proj_size

        if self.proj_size is None:
            self.data_name = name
            self.output_name = self.data_name
        else:
            self.data_name = name + "_pre"
            self.proj_name = name + "_proj"
            self.relu_name = name + "_relu"
            self.output_name = self.relu_name

    @profile
    def forward(self, data):
        assert len(data.shape) == 4
        batch_size, channels, width, height = data.shape
        if self.data_name not in self.apollo_net.blobs.keys():
            self.apollo_net.f(layers.NumpyData(self.data_name, data=data))
        else:
            self.apollo_net.blobs[self.data_name].data[:] = data
        if self.proj_size is not None:
            self.apollo_net.f(layers.Convolution(
                self.proj_name, (1,1), self.proj_size,
                bottoms=[self.data_name]))
            self.apollo_net.f(layers.ReLU(
                self.relu_name, bottoms=[self.proj_name]))

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
        acc = self.apollo_net.f(my_layers.Accuracy(
            self.acc_name, bottoms=[self.output_name, self.target_name]))
        return acc
