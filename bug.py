#!/usr/bin/env python2

import caffe
import apollocaffe
from apollocaffe import ApolloNet, layers
import numpy as np
import timeit

#caffe.set_mode_gpu()
apollocaffe.set_device(0)
net = ApolloNet()
batch_size = 64

data = np.random.random(size=(batch_size, 512, 20, 20)).astype(np.float32)
labels = np.random.randint(10, size=(batch_size,)).astype(np.int32).astype(np.float32)

#print data.dtype
#print labels.dtype

#def load_mem():
#    net.clear_forward()
#    net.f(layers.MemoryData(
#        "mem", data, labels, tops=["input_top", "label_top"],
#        batch_size=batch_size, channels=512, width=20, height=20))
#
#def load_np():
#    net.clear_forward()
#    net.f(layers.NumpyData("np", data))
#
#load_mem()
#load_np()

#data = np.zeros((64, 512, 20, 20))
for i in range(10):
    print
    net.clear_forward()
    import time; s = time.time()
    net.f(layers.NumpyData('a', data))
    print time.time() - s
    import time; s = time.time()
    net.clear_forward()
    net.blobs['a'].data[:] = data
    print time.time() - s

    import time; s = time.time()
    net.clear_forward()
    net.f(layers.MemoryData(
        "b", data, labels, tops=["input_top", "label_top"],
        batch_size=batch_size, channels=512, width=20, height=20))
    print time.time() - s

    

#def prep():
#    net.clear_forward()
#    net.f(layers.NumpyData("input", data=np.random.random(size=(batch_size,512,20,20))))
#
#def load_layer():
#    net.clear_forward()
#    net.blobs["input"].data[...] = np.random.random(size=(batch_size,512,20,20))
#    net.f(layers.InnerProduct("ip", 512, bottoms=["input"]))
#
#def load_sloppy():
#    net.clear_forward()
#    net.f(layers.NumpyData("input", data=np.random.random(size=(batch_size,512,20,20))))
#    net.f(layers.InnerProduct("ip", 512, bottoms=["input"]))
#
#prep()
#
#print "mem", timeit.timeit("load_mem()", number=100, setup="from __main__ import load_mem")
#print "np", timeit.timeit("load_np()", number=100, setup="from __main__ import load_np")

#for i in range(32, 64):
#    load(i)
#    time = timeit.timeit('load(%d)' % i, number=100, setup="from __main__ import load")
#    print time / i

#net.clear_forward()
#net.f(layers.NumpyData("words", data=np.random.randint(10, size=(7,))))
#print net.blobs["words"].shape
#net.f(layers.Wordvec("vecs", 10, 10, bottoms=["words"]))
#print net.blobs["vecs"].shape
#
#net.clear_forward()
#net.f(layers.NumpyData("words", data=np.random.randint(10, size=(14,))))
#print net.blobs["words"].shape
#net.f(layers.Wordvec("vecs", 10, 10, bottoms=["words"]))
#print net.blobs["vecs"].shape

#@profile
#def main():
#    for i in range(1000):
#        net.clear_forward()
#        net.f(layers.NumpyData("input", data=np.random.random((256, 8, 20, 20))))
#        net.f(layers.NumpyData("target", data=np.random.randint(32, size=(256,))))
#        net.f(layers.InnerProduct("ip1", 32, bottoms=["input"]))
#        net.f(layers.SoftmaxWithLoss("loss", bottoms=["ip1", "target"]))
#        net.backward()
#
#        net.clear_forward()
#        net.f(layers.NumpyData("input", data=np.random.random((256, 8, 20, 20))))
#        net.f(layers.NumpyData("target", data=np.random.randint(32, size=(256,))))
#        net.f(layers.InnerProduct("ip2", 32, bottoms=["input"]))
#        net.f(layers.SoftmaxWithLoss("loss2", bottoms=["ip2", "target"]))
#        net.backward()
#
#main()

#print net.blobs["output"].data
#
#net.clear_forward()
#net.f(layers.NumpyData("input", data=np.ones((1, 16, 2, 2))))
#net.f(layers.Convolution("conv2", (1,1), 1, bottoms=["input"]))
#net.f(layers.InnerProduct("output", 8, bottoms=["conv2"]))
#net.backward()
#
#print net.blobs["output"].data
