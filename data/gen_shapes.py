#!/usr/bin/env python2

from util import *

import cairo
import logging
import logging.config
import numpy as np
import yaml

N_QUERY_INSTS = 20

N_TRAIN_TINY  = 10
N_TRAIN_SMALL = 100
N_TRAIN_MED   = 1000
#N_TRAIN_LARGE = 10000
N_TRAIN_ALL   = N_TRAIN_MED

N_TEST        = 1000
N_VAL         = 1000

SHAPE_CIRCLE = 0
SHAPE_SQUARE = 1
SHAPE_TRIANGLE = 2
N_SHAPES = SHAPE_TRIANGLE + 1
SHAPE_STR = {0: "circle", 1: "square", 2: "triangle"}

SIZE_SMALL = 0
SIZE_BIG = 1
N_SIZES = SIZE_BIG + 1
SIZE_STR = {0: "small", 1: "big"}

COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
N_COLORS = COLOR_BLUE + 1
COLOR_STR = {0: "red", 1: "green", 2: "blue"}

WIDTH = 60
HEIGHT = 60

N_CELLS = 3

CELL_WIDTH = WIDTH / N_CELLS
CELL_HEIGHT = HEIGHT / N_CELLS

BIG_RADIUS = CELL_WIDTH * .75 / 2
SMALL_RADIUS = CELL_WIDTH * .5 / 2

GRAMMAR = {
  "$Q": [
    Rule("$Q", ("count", "$S"), 1),
    Rule("$Q", ("shape", "$S"), 2),
    Rule("$Q", ("color", "$S"), 2),
    Rule("$Q", ("size", "$S"), 2),
  ],
  "$S": [
    Rule("$S", ("next_to", "$S"), 1),
    Rule("$S", ("left_of", "$S"), 1),
    Rule("$S", ("right_of", "$S"), 1),
    Rule("$S", ("above", "$S"), 1),
    Rule("$S", ("below", "$S"), 1),
    Rule("$S", ("or", "$S", "$S"), 1),
    Rule("$S", ("and", "$S", "$S"), 1),
    Rule("$S", ("xor", "$S", "$S"), 1),
    Rule("$S", "small", 1),
    Rule("$S", "big", 1),
    Rule("$S", "red", 1),
    Rule("$S", "green", 1),
    Rule("$S", "blue", 1),
    Rule("$S", "circle", 1),
    Rule("$S", "square", 1),
    Rule("$S", "triangle", 1),
    Rule("$S", "nothing", 1)
  ]
}

def draw(shape, color, size, left, top, ctx):
  center_x = (left + .5) * CELL_WIDTH
  center_y = (top + .5) * CELL_HEIGHT

  radius = SMALL_RADIUS if size == SIZE_SMALL else BIG_RADIUS
  radius *= (.9 + np.random.random() * .2)

  if color == COLOR_RED:
    rgb = np.asarray([1., 0., 0.])
  elif color == COLOR_GREEN:
    rgb = np.asarray([0., 1., 0.])
  else:
    rgb = np.asarray([0., 0., 1.])
  rgb += (np.random.random(size=(3,)) * .4 - .2)
  rgb = np.clip(rgb, 0., 1.)

  if shape == SHAPE_CIRCLE:
    ctx.arc(center_x, center_y, radius, 0, 2*np.pi)
    ctx.set_source_rgb(*rgb)
  elif shape == SHAPE_SQUARE:
    ctx.new_path()
    ctx.move_to(center_x - radius, center_y - radius)
    ctx.line_to(center_x + radius, center_y - radius)
    ctx.line_to(center_x + radius, center_y + radius)
    ctx.line_to(center_x - radius, center_y + radius)
  else:
    ctx.new_path()
    ctx.move_to(center_x - radius, center_y + radius)
    ctx.line_to(center_x, center_y - radius)
    ctx.line_to(center_x + radius, center_y + radius)
  ctx.fill()

class Image:
  def __init__(self, shapes, colors, sizes, data):
    self.shapes = shapes
    self.colors = colors
    self.sizes = sizes
    self.data = data

def sample_image():
  data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
  surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
  ctx = cairo.Context(surf)
  ctx.set_source_rgb(1., 1., 1.)
  ctx.paint()

  shapes = [[None for c in range(3)] for r in range(3)]
  colors = [[None for c in range(3)] for r in range(3)]
  sizes = [[None for c in range(3)] for r in range(3)]

  for r in range(3):
    for c in range(3):
      if np.random.random() < 0.2:
        continue
      shape = np.random.randint(N_SHAPES)
      color = np.random.randint(N_COLORS)
      size = np.random.randint(N_SIZES)
      draw(shape, color, size, c, r, ctx)
      shapes[r][c] = shape
      colors[r][c] = color
      sizes[r][c] = size

  surf.write_to_png("_sample.png")
  return Image(shapes, colors, sizes, data)

def evaluate(query, image):
  # forgive me
  if isinstance(query, tuple):
    head = query[0]
    a1 = evaluate(query[1], image)
    a2 = evaluate(query[2], image) if len(query) > 2 else None
    if a1 is None:
      return None
    a1 = list(a1)
    a2 = list(a2) if a2 is not None else None

    if head == "count":
      return len(a1)
    elif head == "shape":
      shape = image.shapes[a1[0][0]][a1[0][1]] if len(a1) == 1 else None
      return SHAPE_STR[shape] if shape is not None else None
    elif head == "color":
      color = image.colors[a1[0][0]][a1[0][1]] if len(a1) == 1 else None
      return COLOR_STR[color] if color is not None else None
    elif head == "size":
      size = image.sizes[a1[0][0]][a1[0][1]] if len(a1) == 1 else None
      return SIZE_STR[size] if size is not None else None
    if head == "next_to":
      # TODO(jda)
      return None
    elif head == "left_of":
      return [(r,c-1) for r,c in a1 if c > 0]
    elif head == "right_of":
      return [(r,c+1) for r,c in a1 if c < N_CELLS-1]
    elif head == "above":
      return [(r-1,c) for r,c in a1 if r > 0]
    elif head == "below":
      return [(r+1,c) for r,c in a1 if r < N_CELLS-1]

    if a2 == None:
      return None

    elif head == "and":
      return set(a1) & set(a2)
    elif head == "or":
      return set(a1) | set(a2)
    elif head == "xor":
      return set(a1) ^ set(a2)

  else:
    if query == "small":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.sizes[r][c] == SIZE_SMALL]
    elif query == "big":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.sizes[r][c] == SIZE_BIG]
    elif query == "red":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.colors[r][c] == COLOR_RED]
    elif query == "green":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.colors[r][c] == COLOR_GREEN]
    elif query == "blue":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.colors[r][c] == COLOR_BLUE]
    elif query == "circle":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.colors[r][c] == SHAPE_CIRCLE]
    elif query == "square":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.colors[r][c] == SHAPE_SQUARE]
    elif query == "triangle":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.colors[r][c] == SHAPE_TRIANGLE]
    elif query == "nothing":
      return [(r,c) for r in range(N_CELLS) for c in range(N_CELLS) if image.shapes[r][c] is None]

  return None

def gen_images(query):
  data = []
  results = set()
  i = 0
  while i < N_QUERY_INSTS * 4 and len(data) < N_QUERY_INSTS:
    i += 1
    image = sample_image()
    result = evaluate(query, image)
    if result is not None and result != 0:
      data.append((query, image, result))
      results.add(result)

  if len(data) == N_QUERY_INSTS and len(results) > 1:
    return data
  else:
    return None

if __name__ == "__main__":
  with open("../log.yaml") as log_config_f:
    logging.config.dictConfig(yaml.load(log_config_f))

  seen = set()
  train_data = []
  val_data = []
  test_data = []

  while len(train_data) < N_TRAIN_ALL * N_QUERY_INSTS:
    query = sample("$Q", GRAMMAR)
    if query not in seen:
      images = gen_images(query)
      if images is not None:
        train_data += images
        seen.add(query)
        if len(train_data) % (1000 * N_QUERY_INSTS) == 0:
          logging.debug("%d / %d", len(train_data) / N_QUERY_INSTS, N_TRAIN_ALL)
  logging.debug("generated train")
  train_data_tiny = train_data[:N_TRAIN_TINY * N_QUERY_INSTS]
  train_data_small = train_data[:N_TRAIN_SMALL * N_QUERY_INSTS]
  train_data_med = train_data[:N_TRAIN_MED * N_QUERY_INSTS]
  train_data_large = train_data

  while len(val_data) < N_VAL * N_QUERY_INSTS:
    query = sample("$Q", GRAMMAR)
    if query not in seen:
      images = gen_images(query)
      if images is not None:
        val_data += images
        seen.add(query)
  logging.debug("generated val")

  while len(test_data) < N_TEST * N_QUERY_INSTS:
    query = sample("$Q", GRAMMAR)
    if query not in seen:
      images = gen_images(query)
      if images is not None:
        test_data += images
        seen.add(query)
  logging.debug("generated test")

  sets = {
    "train.tiny": train_data_tiny,
    "train.small": train_data_small,
    "train.med": train_data_med,
    "train.large": train_data_large,
    "val": val_data,
    "test": test_data
  }

  for set_name, set_data in sets.items():
    with open("shapes/%s.query" % set_name, "w") as query_f, \
         open("shapes/%s.input" % set_name, "w") as input_f, \
         open("shapes/%s.output" % set_name, "w") as output_f:
      for query, image, result in set_data:
        str_query = pp(query)
        print >>query_f, str_query
        print >>output_f, result

        image_data = image.data
        image_data = image_data[:,:,0:3].flatten().tolist()
        print >>input_f, " ".join([str(v) for v in image_data])
