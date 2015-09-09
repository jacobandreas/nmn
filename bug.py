#!/usr/bin/env python2

import caffe
from apollocaffe import ApolloNet, layers
import numpy as np

#caffe.set_mode_gpu()
net = ApolloNet()

@profile
def main():
    for i in range(1000):
        net.clear_forward()
        net.f(layers.NumpyData("input", data=np.random.random((256, 8, 20, 20))))
        net.f(layers.NumpyData("target", data=np.random.randint(32, size=(256,))))
        net.f(layers.InnerProduct("ip1", 32, bottoms=["input"]))
        net.f(layers.SoftmaxWithLoss("loss", bottoms=["ip1", "target"]))
        net.backward()

        net.clear_forward()
        net.f(layers.NumpyData("input", data=np.random.random((256, 8, 20, 20))))
        net.f(layers.NumpyData("target", data=np.random.randint(32, size=(256,))))
        net.f(layers.InnerProduct("ip2", 32, bottoms=["input"]))
        net.f(layers.SoftmaxWithLoss("loss2", bottoms=["ip2", "target"]))
        net.backward()

main()


#print net.blobs["output"].data
#
#net.clear_forward()
#net.f(layers.NumpyData("input", data=np.ones((1, 16, 2, 2))))
#net.f(layers.Convolution("conv2", (1,1), 1, bottoms=["input"]))
#net.f(layers.InnerProduct("output", 8, bottoms=["conv2"]))
#net.backward()
#
#print net.blobs["output"].data
