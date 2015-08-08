#!/usr/bin/env python2

import corpus
import nmn as mod_nmn
from input_types import ImageInput
from output_types import ImageOutput

import numpy as np
import theano
import theano.tensor as T
from lasagne import layers, nonlinearities, objectives, updates
import sys

if __name__ == "__main__":

  l_input = layers.InputLayer((256, 3, 30, 30))
  
  #l_conv1_a = layers.Conv2DLayer(l_input, 8, 5, border_mode="same")
  #l_pool1_a = layers.MaxPool2DLayer(l_conv1_a, 2)
  #l_drop1_a = layers.DropoutLayer(l_pool1_a, p=0.5)
  #l_conv2_a = layers.Conv2DLayer(l_drop1_a, 1, 3, border_mode="same")
  #l_pool2_a = layers.MaxPool2DLayer(l_conv2_a, 2)
  l_pool2_a = layers.DenseLayer(l_input, 32)

  #l_conv1_b = layers.Conv2DLayer(l_input, 8, 5, border_mode="same")
  #l_pool1_b = layers.MaxPool2DLayer(l_conv1_b, 2)
  #l_drop1_b = layers.DropoutLayer(l_pool1_b, p=0.5)
  #l_conv2_b = layers.Conv2DLayer(l_drop1_b, 1, 3, border_mode="same")
  #l_pool2_b = layers.MaxPool2DLayer(l_conv2_b, 2)
  l_pool2_b = layers.DenseLayer(l_input, 32)

  l_merge = layers.ElemwiseMergeLayer([l_pool2_a, l_pool2_b], T.minimum)

  #l_drop1 = layers.DropoutLayer(l_merge, p=0.5)
  l_dense1 = layers.DenseLayer(l_merge, 32)
  #l_drop2 = layers.DropoutLayer(l_dense1, p=0.5)
  l_dense2 = layers.DenseLayer(l_dense1, 2, nonlinearity=nonlinearities.softmax)

  l_out = l_dense2

  params = layers.get_all_params(l_out)
  t_input = T.tensor4("input")
  t_output = layers.get_output(l_out, t_input)
  t_output_det = layers.get_output(l_out, t_input, deterministic=True)
  t_target = T.ivector("output")
  t_pred = T.argmax(t_output_det, axis=1)

  t_loss = objectives.aggregate(objectives.categorical_crossentropy(t_output,
      t_target))

  #upd = updates.nesterov_momentum(t_loss, params, 0.001)
  upd = updates.adadelta(t_loss, params)
  #upd = updates.adam(t_loss, params)

  train = theano.function([t_input, t_target], t_loss, updates=upd)
  predict = theano.function([t_input], t_pred)

  all_data = corpus.load("shapes", "train.large")[0*2048:1*2048]
  data = all_data[:-256]
  val_data = all_data[-256:]

  for i in range(40000):
    np.random.shuffle(data)
    batch = data[:256]

    #for datum in batch:
    #  print "==="
    #  print datum.query
    #  print datum.output
    #  for channel in range(3):
    #    for r in range(30):
    #      for c in range(30):
    #        mark = "#" if datum.input_[channel, r, c] > 0.5 else "."
    #        print mark,
    #      print
    #    print
    #exit()

    batch_inputs = np.asarray([datum.input_ for datum in batch], dtype=theano.config.floatX)
    batch_outputs = np.asarray([datum.output for datum in batch], dtype=np.int32)
    err = train(batch_inputs, batch_outputs)
    pred = predict(batch_inputs)
    acc = 1. * sum(np.equal(pred, batch_outputs)) / len(batch)
    print err, acc,

    val_inputs = np.asarray([datum.input_ for datum in val_data], dtype=theano.config.floatX)
    val_outputs = np.asarray([datum.output for datum in val_data], dtype=np.int32)
    val_pred = predict(val_inputs)
    val_acc = 1. * sum(np.equal(val_pred, val_outputs)) / len(val_data)
    print val_acc

    if i % 100 == 0:
      print "saved"
      #np.save("weights/filters_a", l_conv1_a.W.get_value())
      #np.save("weights/filters_b", l_conv1_b.W.get_value())
