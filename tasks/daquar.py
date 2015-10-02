#!/usr/bin/env python2

from datum import Datum, Layout
from indices import STRING_INDEX, LAYOUT_INDEX, ANSWER_INDEX
from parse import parse_tree
from models.modules import AttAnswerModule, IndexedConvModule, DenseAnswerModule, ConjModule

from collections import defaultdict
import logging
import numpy as np

STRING_FILE = "data/daquar/%s/%s.questions.txt"
PARSE_FILE = "data/daquar/%s/%s.questions.sp"
ANN_FILE = "data/daquar/%s/%s.answers.txt"
IMAGE_FILE = "data/daquar/images/conv/%d.png.npz"
RAW_IMAGE_FILE = "data/daquar/images/raw/%d.png"

def parse_to_layout(parse):
    return Layout(*parse_to_layout_helper(parse), internal=False)

def parse_to_layout_helper(parse, internal):
    if isinstance(parse, str):
        return (IndexedConvModule, LAYOUT_INDEX.index(parse))
    else:
        head = parse[0]
        if internal:
            if head == "and":
                return ConjModule
            else:
                return RedetectModule
        else:
            if head in ("how many"):
                return DenseAnswerModule
            else:
                return AttAnswerModule
        if head[
        if parse[0] == 
        print parse[0]

class DaquarTask:
    def __init__(self, config):
        self.train = DaquarTaskSet(config, "train")
        self.val = DaquarTaskSet(config, "val")
        self.test = DaquarTaskSet(config, "test")

class DaquarTaskSet:
    def __init__(self, config, set_name):
        self.config = config

        size = config.train_size

        data_by_id = dict()
        data_by_layout_type = defaultdict(list)
        data_by_sentence_length = defaultdict(list)

        if set_name == "val":
            self.by_id = data_by_id
            self.by_layout_type = data_by_layout_type
            self.by_sentence_length = data_by_sentence_length
            self.layout_types = set()
            self.sentence_lengths = set()
            return

        with open(STRING_FILE % (size, set_name)) as question_f, \
             open(PARSE_FILE % (size, set_name)) as parse_f, \
             open(ANN_FILE % (size, set_name)) as ann_f:

           for question, parse_str, answer in zip(question_f, parse_f, ann_f):
               question = question.strip()
               parse_str = parse_str.strip()
               answer = answer.strip()
               words = question.split()
               image_id = words[-2]
               words = ["<s>"] + words[:-4] + ["</s>"]

               indexed_words = [STRING_INDEX.index(w) for w in words]

               print answer

               parse = parse_tree(parse_str)
               layout = parse_to_layout(parse)
               datum = DaquarDatum(indexed_words, layout, image_id, answer)
