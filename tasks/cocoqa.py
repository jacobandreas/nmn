#!/usr/bin/env python2

from datum import Datum, Layout
from indices import STRING_INDEX, LAYOUT_INDEX, ANSWER_INDEX
from parse import parse_tree
from models.modules import \
        AttAnswerModule, DetectModule, DenseAnswerModule, ConjModule, \
        RedetectModule

from collections import defaultdict
import logging
import numpy as np

STRING_FILE = "data/cocoqa/%s/questions.txt"
PARSE_FILE = "data/cocoqa/%s/questions.sp"
ANN_FILE = "data/cocoqa/%s/answers.txt"
IMAGE_ID_FILE = "data/cocoqa/%s/img_ids.txt"
IMAGE_FILE = "data/images/Images/%s2014/conv/COCO_%s2014_%012d.jpg.npz"
RAW_IMAGE_FILE = "data/images/Images/%s2014/COCO_%s2014_%012d.jpg"

def parse_to_layout(parse):
    return Layout(*parse_to_layout_helper(parse, internal=False))

def parse_to_layout_helper(parse, internal):
    if isinstance(parse, str):
        return (DetectModule, LAYOUT_INDEX.index(parse))
    else:
        head = parse[0]
        head_idx = LAYOUT_INDEX.index(parse)
        if internal:
            if head == "and":
                mod_head = ConjModule
            else:
                mod_head = RedetectModule
        else:
            if head == "count":
                mod_head = DenseAnswerModule
            else:
                mod_head = AttAnswerModule

        below = [parse_to_layout_helper(child, internal=True) for child in parse[1:]]
        mods_below, indices_below = zip(*below)
        return (mod_head,) + tuple(mods_below), (head_idx,) + tuple(indices_below)

class CocoQADatum(Datum):
    def __init__(self, string, layout, image_id, answer, coco_set_name):
        self.string = string
        self.layout = layout
        self.image_id = image_id
        self.answer = answer
        self.outputs = [answer]

        self.input_path = IMAGE_FILE % (coco_set_name, coco_set_name, image_id)
        self.image_path = RAW_IMAGE_FILE % (coco_set_name, coco_set_name, image_id)

    def load_input(self):
        with np.load(self.input_path) as zdata:
            assert len(zdata.keys()) == 1
            image_data = zdata[zdata.keys()[0]]
        return image_data

class CocoQATask:
    def __init__(self, config):
        self.train = CocoQATaskSet(config, "train")
        self.val = CocoQATaskSet(config, "val")
        self.test = CocoQATaskSet(config, "test")


class CocoQATaskSet:
    def __init__(self, config, set_name):
        self.config = config

        data = set()
        data_by_layout_type = defaultdict(list)
        data_by_string_length = defaultdict(list)
        data_by_layout_and_length = defaultdict(list)

        if set_name == "val":
            self.data = data
            self.by_layout_type = data_by_layout_type
            self.by_string_length = data_by_string_length
            self.by_layout_and_length = data_by_layout_and_length
            return


        with open(STRING_FILE % set_name) as question_f, \
             open(PARSE_FILE % set_name) as parse_f, \
             open(ANN_FILE % set_name) as ann_f, \
             open(IMAGE_ID_FILE % set_name) as image_id_f:

            i = 0
            for question, parse_str, answer, image_id in zip(question_f, parse_f, ann_f, image_id_f):

                #if i > 30000:
                #    break
                i += 1
            
                question = question.strip()
                parse_str = parse_str.strip().replace("'", "")
                answer = answer.strip()
                image_id = int(image_id.strip())
                words = question.split()
                words = ["<s>"] + words + ["</s>"]

                answer = ANSWER_INDEX.index(answer)
                words = [STRING_INDEX.index(w) for w in words]
                parse = parse_tree(parse_str)
                layout = parse_to_layout(parse)

                if parse[0] != "color":
                    continue

                coco_set_name = "train" if set_name == "train" else "val"
                datum = CocoQADatum(words, layout, image_id, answer, coco_set_name)

                data.add(datum)
                data_by_layout_type[datum.layout.modules].append(datum)
                data_by_string_length[len(datum.string)].append(datum)
                data_by_layout_and_length[(datum.layout.modules, len(datum.string))].append(datum)

        self.data = data
        self.by_layout_type = data_by_layout_type
        self.by_string_length = data_by_string_length
        self.by_layout_and_length = data_by_layout_and_length

        logging.info("%s:", set_name.upper())
        logging.info("%s items", len(self.data))
        logging.info("%s words", len(STRING_INDEX))
        logging.info("%s functions", len(LAYOUT_INDEX))
        logging.info("%s answers", len(ANSWER_INDEX))
        logging.info("%s layouts", len(self.by_layout_type.keys()))
        logging.info("")
