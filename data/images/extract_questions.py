#!/usr/bin/env python2

import json
import sys

with open(sys.argv[1]) as data_file, \
     open("ids.txt", "w") as id_file, \
     open("questions.txt", "w") as question_file:
  data = json.load(data_file)
  questions = data["questions"]
  for question in questions:
    image_id = question["image_id"]
    question_id = question["question_id"]
    print >>id_file, "%s,%s" % (question_id, image_id)
    print >>question_file, question["question"].encode("ascii", "ignore")
