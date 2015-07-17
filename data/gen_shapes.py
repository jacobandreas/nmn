#!/usr/bin/env python2

from util import *

import cairo
import numpy as np

SHAPE_CIRCLE = 0

SIZE_SMALL = 0
SIZE_BIG = 1

COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2

WIDTH = 60
HEIGHT = 60

N_CELLS = 3

CELL_WIDTH = WIDTH / N_CELLS
CELL_HEIGHT = HEIGHT / N_CELLS

BIG_RADIUS = CELL_WIDTH * .75
SMALL_RADIUS = CELL_WIDTH * .5

def draw(shape, color, size, left, top, ctx):
  center_x = (left + .5) * CELL_WIDTH
  center_y = (top + .5) * CELL_HEIGHT

if __name__ == "__main__":
  data = np.zeros((200, 200, 4), dtype=np.uint8)
  surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, 200, 200)
  cr = cairo.Context(surf)

  cr.set_source_rgb(1., 1., 1.)
  cr.paint()

  cr.arc(100, 100, 80, 0, 2 * np.pi)
  cr.set_line_width(3)
  cr.set_source_rgb(1., 0., 0.)
  cr.stroke()

  print data[38:48,38:48,0]
