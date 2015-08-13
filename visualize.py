#!/usr/bin/env python2

import os
import cairo
import numpy as np


scale = 10


for filter_fname in os.listdir("weights"):
  filter_name = filter_fname[:-3]
  filter_ext = filter_fname[-3:]
  if filter_ext != "npy": continue
  filter_path = os.path.join("weights", filter_fname)
  filter = np.load(filter_path)

  #print filter.shape
  #if filter.shape[2] != 5: continue
  in_width = filter.shape[2]
  in_height = filter.shape[3]

  out_width = scale * in_width #50
  out_height = scale * in_height #50

  #scaled_filter = np.kron(filter, np.ones((scale,scale)))

  fmax = max(np.max(filter), abs(np.min(filter)))
  print fmax

  for i_filter in range(filter.shape[0]):
    data = np.zeros((out_width, out_height, 4), dtype=np.uint8)
    surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, out_width, out_height)
    data[:,:,3] = 255

    for i_channel in range(3):
      i_filter_channel = i_channel % filter.shape[1]
      #for c in range(3):
      #  sc = 2. if c == i_channel else 1.
      #  data[:,:,c] = 128 + sc * 60 * scaled_filter[i_filter,i_filter_channel,:,:]
      #print filter[i_filter,i_filter_channel,:,:]
      scaled_filter = np.kron(filter[i_filter, i_filter_channel,:,:], np.ones((scale, scale)))
      scaled_filter = 128 * scaled_filter / fmax
      data[:,:,2-i_channel] = 128 + scaled_filter

    surf.write_to_png("weights/%s_%d.png" % (filter_fname, i_filter))

#
#
#surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, width, height)
#ctx = cairo.Context(surf)
#
#data[:,:,:] = 255
#data[:,:,0] = 0
#
