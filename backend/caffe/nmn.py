#!/usr/bin/env python2

import module

import apollocaffe
import itertools
import importlib
import numpy as np

class ModuleNetwork:
    def __init__(self, query, model):
        self.input_module = model.get_input_module()
        self.support_module = model.get_support_module()

        self.modules = []
        self.wire(self.modules, query, model)

        self.target_module = model.get_target_module()
        self.loss_module = model.get_loss_module(self.modules[-1].last_layer_name)
        self.eval_module = model.get_eval_module(self.modules[-1].last_layer_name)

    def wire(self, modules, query, model):
        if not isinstance(query, tuple):
            module = model.get_module(
                    query, 0, self.input_module.last_layer_name, [])
            modules.append(module)
        else:
            head = query[0]
            tail = query[1:]
            arity = len(tail)

            children = [self.wire(modules, tail_query, model) 
                        for tail_query in tail]

            module = model.get_module(query[0], 
                                      arity,
                                      self.input_module.last_layer_name,
                                      children)
            modules.append(module)

        return module.last_layer_name

    def forward(self, input, target):
        self.input_module.forward(input)
        self.support_module.forward()

        for module in self.modules:
            module.forward()

        self.target_module.forward(target)
        loss = self.loss_module.forward(target)
        eval = self.eval_module.forward(target)
        return loss, eval

class NMNModel:
    def __init__(self, config, opt_config):
        self.config = config
        self.opt_config = opt_config

        self.module_builder = importlib.import_module(
                "task.%s" % config.module_builder.name)

        self.nets = dict()
        self.current_net = None

        self.apollo_net = apollocaffe.ApolloNet()

        self.sq_grads = dict()
        self.sq_updates = dict()

        self.loaded = False

    def forward(self, query, input, target):
        assert self.current_net is None
        self.apollo_net.clear_forward()
        self.current_net = self.get_net(query)
        return self.current_net.forward(input, target)

    def train(self):
        assert self.current_net is not None
        self.apollo_net.backward()
        #print self.apollo_net.blobs["Answer_color__ip"].data
        #print self.apollo_net.params["Answer_color__ip.p0"].data
        #print np.sum(self.apollo_net.params["Answer_color__ip.p0"].data)
        #print np.sum(np.square(self.apollo_net.params["Answer_color__ip.p0"].data))
        #self.update()

        self.apollo_net.update(lr=self.opt_config.learning_rate,
                               momentum=self.opt_config.momentum,
                               clip_gradients=self.opt_config.clip)
        return

        if not self.loaded:
            self.loaded = True
            self.apollo_net.params["Answer_color__ip.p0"].data[...] = \
                np.load("../nmn/weights/_output-1.w.npy").T
            self.apollo_net.params["Answer_color__ip.p1"].data[...] = \
                np.load("../nmn/weights/_output-1.b.npy")
            self.apollo_net.params["IndexedConv__conv1.0"].data[...] = \
                np.load("../nmn/weights/conv_with_embedding.w_hidden.npy").T.reshape((64,356,1,1))
            self.apollo_net.params["IndexedConv__conv1.1"].data[...] = \
                np.load("../nmn/weights/conv_with_embedding.b_hidden.npy")
            self.apollo_net.params["IndexedConv__conv2.0"].data[...] = \
                np.load("../nmn/weights/conv_with_embedding.w_out.npy").T.reshape((1,64,1,1))
            self.apollo_net.params["IndexedConv__conv2.1"].data[...] = \
                np.load("../nmn/weights/conv_with_embedding.b_out.npy")
        else:
            #pass
            self.update()

    def update(self):
        rho = self.opt_config.rho
        epsilon = self.opt_config.epsilon
        lr = self.opt_config.lr
        clip = self.opt_config.clip
        for param_name in self.apollo_net.active_param_names():
            param = self.apollo_net.params[param_name]
            grad = param.diff

            if param_name in self.sq_grads:
                self.sq_grads[param_name] = \
                    (1 - rho) * np.square(grad) + rho * self.sq_grads[param_name]
                rms_update = np.sqrt(self.sq_updates[param_name] + epsilon)
                rms_grad = np.sqrt(self.sq_grads[param_name] + epsilon)
                update = -rms_update / rms_grad * grad
                self.sq_updates[param_name] = \
                    (1 - rho) * np.square(update) + rho * self.sq_updates[param_name]
            else:
                self.sq_grads[param_name] = (1 - rho) * np.square(grad)
                update = np.sqrt(epsilon + self.sq_grads[param_name])
                self.sq_updates[param_name] = (1 - rho) * np.square(update)

            update *= lr

            norm = np.sqrt(np.sum(np.square(update)))
            if norm > clip:
                update = clip * update / norm
            
            param.data[...] += update

    def clear(self):
        self.current_net = None

    def get_net(self, query):
        if query not in self.nets:
            net = ModuleNetwork(query, self)
            self.nets[query] = net
        return self.nets[query]

    def get_module(self, name, arity, input_name, incoming_names):
        return self.module_builder.build_caffe_module(
                name, arity, input_name, incoming_names, self.apollo_net,
                self.config.module_builder)

    def get_input_module(self):
        return module.DataModule("Input", self.apollo_net)

    def get_support_module(self):
        return self.module_builder.build_caffe_support_module(
                self.apollo_net, self.config.module_builder)

    def get_target_module(self):
        return module.DataModule("Target", self.apollo_net)

    def get_loss_module(self, output_name):
        return self.module_builder.build_caffe_loss_module(
                output_name, self.apollo_net)

    def get_eval_module(self, output_name):
        return self.module_builder.build_caffe_eval_module(
                output_name, self.apollo_net)
