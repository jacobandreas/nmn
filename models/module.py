#!/usr/bin.env python2

import my_layers

from apollocaffe import layers
import numpy as np

class NullModule:
    def forward(self):
        pass

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
        self.apollo_net = apollo_net

        self.conv_name = "Conv_%s__conv" % name
        self.relu_name = "Conv_%s__relu" % name
        self.last_layer_name = self.relu_name

    def forward(self):
        self.apollo_net.f(layers.Convolution(
            self.conv_name, (1,1), 1, bottoms=[self.input_name]))

        self.apollo_net.f(layers.ReLU(self.relu_name, bottoms=[self.conv_name]))

class IndexedConvModule:
    def __init__(self, name, embeddings, embedding_index, hidden_size,
                 input_name, incoming_names, apollo_net):
        self.name = name
        self.hidden_size = hidden_size
        self.input_name = input_name
        assert len(incoming_names) == 0
        self.apollo_net = apollo_net

        if name in embedding_index:
            word_index = embedding_index[name]
        else:
            word_index = embedding_index["*unknown*"]
        self.embedding = embeddings[word_index,:].reshape((1,-1))

        self.vector_name = "IndexedConv_%s__vec" % name
        #self.proj_name = "IndexedConv_%s__proj" % name
        self.proj_name = "IndexedConv__proj"

        self.conv1_name = "IndexedConv__conv1"
        self.relu1_name = "IndexedConv__relu1"
        self.bc_sum_name = "IndexedConv__bc_sum"
        self.conv2_name = "IndexedConv__conv2" 
        self.relu2_name = "IndexedConv__relu2"
        self.last_layer_name = self.relu2_name

        self.conv1_param0_name = "IndexedConv__conv1.0"
        self.conv1_param1_name = "IndexedConv__conv1.1"
        self.conv2_param0_name = "IndexedConv__conv2.0"
        self.conv2_param1_name = "IndexedConv__conv2.1"

    @profile
    def forward(self):
        batch_size, channels, width, height = self.apollo_net.blobs[self.input_name].shape
        #input_shape = str(self.apollo_net.blobs[self.input_name].data.shape)

        self.apollo_net.f(layers.NumpyData(
            self.vector_name, self.embedding))

        self.apollo_net.f(layers.InnerProduct(
            self.proj_name, self.hidden_size, bottoms=[self.vector_name]))

        ## self.apollo_net.blobs[self.vector_name].reshape((1,100,1,1))
        ## self.apollo_net.f(layers.Tile(
        ##     "t1_" + input_shape, bottoms=[self.vector_name], axis=2,
        ##     tiles=self.apollo_net.blobs[self.input_name].shape[2]))
        ## self.apollo_net.f(layers.Tile(
        ##     "t2_" + input_shape, bottoms=["t1_" + input_shape], axis=3,
        ##     tiles=self.apollo_net.blobs[self.input_name].shape[3]))

        #print self.apollo_net.blobs["t2_" + self.name].data.shape
        #print self.apollo_net.blobs[self.input_name].data.shape

        ## self.apollo_net.f(layers.Concat(
        ##     "cat_" + self.name, bottoms=["t2_" + input_shape, self.input_name], axis=1))

        self.apollo_net.f(layers.Convolution(
            #self.conv1_name, (1,1), self.hidden_size, bottoms=["cat_" + self.name],
            self.conv1_name, (1,1), self.hidden_size, bottoms=[self.input_name],
            param_names=[self.conv1_param0_name, self.conv1_param1_name]))

        ### self.apollo_net.f(my_layers.BroadcastSum(
        ###     self.bc_sum_name, bottoms=[self.proj_name, self.conv1_name]))

        t_batch_name = "t_batch_%d" % batch_size
        t_width_name = "t_width_%d" % width
        t_height_name = "t_height_%d" % height

        self.apollo_net.blobs[self.proj_name].reshape((1, self.hidden_size, 1, 1))
        self.apollo_net.f(layers.Tile(
            t_batch_name, bottoms=[self.proj_name], axis=0, tiles=batch_size))
        self.apollo_net.f(layers.Tile(
            t_width_name, bottoms=[t_batch_name], axis=2, tiles=width))
        self.apollo_net.f(layers.Tile(
            t_height_name, bottoms=[t_width_name], axis=3, tiles=height))

        self.apollo_net.f(layers.Eltwise(
            self.bc_sum_name, bottoms=[t_height_name, self.conv1_name],
            operation="SUM"))

        self.apollo_net.f(layers.ReLU(
            self.relu1_name, bottoms=[self.bc_sum_name]))

        self.apollo_net.f(layers.Convolution(
            self.conv2_name, (1,1), 1, bottoms=[self.relu1_name],
            param_names=[self.conv2_param0_name, self.conv2_param1_name]))

        self.apollo_net.f(layers.ReLU(
            self.relu2_name, bottoms=[self.conv2_name]))

# class EmbeddingModule:
#     def __init__(self, embeddings, index, apollo_net):
#         self.embeddings = embeddings
#         self.apollo_net = apollo_net
# 
#         self.data_layer_name = "Embedding__data"
# 
#     def forward(self):
#         self.apollo_net.f(layers.NumpyData(
#                 self.data_layer_name, self.embeddings))
# 
#     def load_embeddings(self, embedding_path):
#         embeddings = dict()
#         with open(embedding_path) as embedding_f:
#             for line in embedding_f:
#                 word, svec = line.split("\t")
#                 vec = np.asarray([float(v) for v in svec.split()])
#                 embeddings[word] = vec
#         return embeddings

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

    @profile
    def forward(self):
        input_channels = self.apollo_net.blobs[self.input_name].shape[1]
        incoming_shape = tuple(self.apollo_net.blobs[self.incoming_names[0]].shape)
        assert incoming_shape[1] == 1
        flat_shape = (incoming_shape[0], incoming_shape[2] * incoming_shape[3])

        # TODO(jda) is this evil?
        self.apollo_net.blobs[self.incoming_names[0]].reshape(flat_shape)

        self.apollo_net.f(layers.Softmax(
                self.softmax_layer_name,
                bottoms=[self.incoming_names[0]]))

        self.apollo_net.blobs[self.softmax_layer_name].reshape(incoming_shape)

        self.apollo_net.f(my_layers.Attention(
                self.attention_layer_name,
                bottoms=[self.softmax_layer_name, self.input_name]))

        self.apollo_net.f(layers.InnerProduct(
                self.ip_layer_name,
                1000,
                bottoms=[self.attention_layer_name]))

class DataModule:
    def __init__(self, name, apollo_net):
        self.apollo_net = apollo_net
        self.last_layer_name = name

    @profile
    def forward(self, data):
        self.apollo_net.f(layers.NumpyData(self.last_layer_name, data=data))

class ClassificationLogLossModule:
    def __init__(self, output_name, apollo_net):
        self.apollo_net = apollo_net
        self.output_name = output_name
        self.target_name = "Target"
        self.loss_name = "Loss__" + output_name

    @profile
    def forward(self, target):
        loss = self.apollo_net.f(layers.SoftmaxWithLoss(
            self.loss_name, bottoms=[self.output_name, self.target_name],
            normalize=False))
        return loss

class ClassificationAccuracyModule:
    def __init__(self, output_name, apollo_net):
        self.apollo_net = apollo_net
        self.output_name = output_name
        self.target_name = "Target"
        self.acc_name = "Accuracy"

    @profile
    def forward(self, target):
        #print self.apollo_net.blobs[self.output_name].data
        return self.apollo_net.f(my_layers.Accuracy(
            self.acc_name, bottoms=[self.output_name, self.target_name]))
