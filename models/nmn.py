#!/usr/bin/env python2

import modules
import util
from visualizer import visualizer

import apollocaffe
import itertools
import importlib
import numpy as np

# TODO consolidate with wire()
def linearize(lin, indices):
    if isinstance(indices, tuple):
        for i in indices[1:]:
            linearize(lin, i)
        lin.append(indices[0])
    else:
        lin.append(indices)

class ModuleNetwork:
    def __init__(self, query, model, include_reading_module=False):
        self.input_module = model.get_input_module()
        self.support_module = model.get_support_module()

        self.modules = []
        self.wire(self.modules, query, model)

        output_name = self.modules[-1].output_name
        if include_reading_module:
            self.reading_module = model.get_reading_module(output_name)
            output_name = self.reading_module.output_name
        else:
            self.reading_module = None

        self.target_module = model.get_target_module()
        self.loss_module = model.get_loss_module(output_name)
        self.eval_module = model.get_eval_module(output_name)

    def wire(self, modules, query, model):
        if not isinstance(query, tuple):
            module = model.get_module(
                    query, 0, self.input_module.output_name, [])
            modules.append(module)
        else:
            head = query[0]
            tail = query[1:]
            arity = len(tail)

            children = [self.wire(modules, tail_query, model) 
                        for tail_query in tail]

            module = model.get_module(query[0], 
                                      arity,
                                      self.input_module.output_name,
                                      children)
            modules.append(module)

        return module.output_name

    def forward(self, indices, input, target, compute_eval):
        self.input_module.forward(input)
        self.support_module.forward()

        for module, mod_indices in zip(self.modules, indices):
            module.forward(mod_indices)

        if self.reading_module is not None:
            self.reading_module.forward(indices)

        self.target_module.forward(target)
        loss = self.loss_module.forward(target)
        if compute_eval:
            eval = self.eval_module.forward(target)
            return loss, eval
        else:
            return loss, None

class NMNModel:
    def __init__(self, config, opt_config):
        self.config = config
        self.opt_config = opt_config

        self.nets = dict()
        self.current_net = None

        self.apollo_net = apollocaffe.ApolloNet()

        self.sq_grads = dict()
        self.sq_updates = dict()

    def forward(self, layout_type, indices, input, target, compute_eval=False):
        assert self.current_net is None
        self.apollo_net.clear_forward()
        self.current_net = self.get_net(layout_type)
        lin_indices = []
        linearize(lin_indices, indices)
        return self.current_net.forward(lin_indices, input, target, compute_eval)

    def train(self):
        assert self.current_net is not None
        self.apollo_net.backward()
        self.apollo_net.update(lr=self.opt_config.learning_rate,
                               momentum=self.opt_config.momentum,
                               clip_gradients=self.opt_config.clip)

    def update(self):
        rho = self.opt_config.rho
        epsilon = self.opt_config.epsilon
        lr = self.opt_config.lr
        clip = self.opt_config.clip

        all_norm = 0.
        for param_name in self.apollo_net.active_param_names():
            param = self.apollo_net.params[param_name]
            grad = param.diff
            all_norm += np.sum(np.square(grad))
        all_norm = np.sqrt(all_norm)

        for param_name in self.apollo_net.active_param_names():
            param = self.apollo_net.params[param_name]
            grad = param.diff

            if all_norm > clip:
                grad = clip * grad / all_norm

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
                update = np.sqrt(epsilon) / np.sqrt(epsilon +
                        self.sq_grads[param_name]) * grad
                self.sq_updates[param_name] = (1 - rho) * np.square(update)

            #print np.sum(np.square(update))
            #print param.data.shape
            #print update.shape
            
            param.data[...] += update
            #print np.sqrt(np.sum(np.square(param.data)))
        #print

    def clear(self):
        self.current_net = None

    def get_net(self, query):
        if query not in self.nets:
            net = ModuleNetwork(query, self, self.config.include_reading_module)
            self.nets[query] = net
        return self.nets[query]

    def get_module(self, module, arity, input_name, incoming_names):
        if module == modules.IndexedConvModule:
            assert len(incoming_names) == 0
            return module(self.config.hidden_size, input_name, self.apollo_net)
        elif module == modules.AttAnswerModule:
            return module(self.config.hidden_size, input_name, incoming_names, self.apollo_net)
        else:
            raise NotImplementedError()

    def get_input_module(self):
        return modules.DataModule("Input", self.apollo_net)

    def get_support_module(self):
        return modules.NullModule()

    def get_target_module(self):
        return modules.DataModule("Target", self.apollo_net)

    def get_loss_module(self, output_name):
        return modules.ClassificationLogLossModule(
                output_name, self.apollo_net)

    def get_eval_module(self, output_name):
        return modules.ClassificationAccuracyModule(
                output_name, self.apollo_net)

    def get_reading_module(self, output_name):
        return modules.LSTMModule(
                self.config.hidden_size, output_name, self.apollo_net)
