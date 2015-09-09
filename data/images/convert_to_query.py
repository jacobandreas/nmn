#!/usr/bin/env python2

import re
import sys

def name_to_match(name, used_names):
  if isinstance(name, int) and name not in used_names:
    r_name = "(?P<g%s>[\\w-]+)" % name
    used_names.add(name)
  elif isinstance(name, int):
    r_name = "(?P=g%s)" % name
  else:
    r_name = name
  return r_name

def render(spec, match):
  if isinstance(spec, str):
    return spec
  if isinstance(spec, int):
    token = match.group("g%d" % spec)
    word = token.split("-")[0]
    return word

  return "(%s)" % " ".join([render(p, match) for p in spec])

def match_simple_query(query, edges):
  edge_spec, query_spec = query
  used_names = set()
  query_re = ".*"
  for rel, head, tail in edge_spec:
    r_head = name_to_match(head, used_names)
    r_tail = name_to_match(tail, used_names)
    query_re += "%s\(%s, %s\).*" % (rel, r_head, r_tail)

  m = re.match(query_re, edges.lower())
  if m is None:
    return None

  return render(query_spec, m)

is_prep_query = (
  [ 
    ("cop", 0, "be-\\d"),
    ("(amod|nsubj)", 0, 1) ,
    ("advmod", 0, 2)
  ],
  ("is", ("and", (2, 0), 1))
)

is_query = (
  [ 
    ("cop", 0, "be-\\d"),
    ("(amod|nsubj)", 0, 1) 
  ],
  ("is", ("and", 0, 1))
)

is_query_2 = (
  [
    ("aux", 0, "be-\\d"),
    ("nsubj", 0, 1)
  ],
  ("is", ("and", 0, 1))
)

whatx_query = (
  [
    ("det", 0, "what-\\d"),
    ("dep", "be-\\d", 0),
    ("nsubj", "be-\\d", 1)
  ],
  (0, 1)
)

whatx_prep_query = (
  [
    ("det", 0, "what-\\d"),
    ("nsubj", "be-\\d", 0),
    ("case", 1, 2),
    ("nmod:\\w+", "be-\\d", 1)
  ],
  (0, (2, 1))
)

whatx_of_query = (
  [
    ("cop", "what-\\d", "be-\\d"),
    ("nsubj", "what-\\d", 0),
    ("nmod:\\w+", 0, 1),
  ],
  (0, 1)
)

wh_prep_query = (
  [
    ("nsubj", "be-\\d", 0),
    ("case", 1, 2),
    ("nmod:\\w+", "be-\\d", 1)
  ],
  (0, (2, 1))
)

wh_query = (
  [
    ("advmod", "be-\\d", 0),
    ("nsubj", "be-\\d", 1)
  ],
  (0, 1)
)

wh_query_2 = (
  [
    ("cop", 0, "be-\\d"),
    ("nsubj", 0, 1)
  ],
  (0, 1)
)

count_query = (
  [
    ("advmod", "many-2", "how-1"),
    ("(amod|dep)", 0, "many-2")
  ],
  ("count", 0)
)

expl_query = (
  [
    ("expl", "be-\\d", "there-\\d"),
    ("nsubj", "be-\\d", 0)
  ],
  ("is", 0)
)

SIMPLE_QUERIES = [
  count_query,
  whatx_prep_query,
  whatx_of_query,
  whatx_query,
  wh_prep_query,
  wh_query,
  #wh_query_2,
  is_prep_query,
  is_query,
  is_query_2,
  expl_query
]
def make_simple_query(edges):
  for query in SIMPLE_QUERIES:
    m = match_simple_query(query, edges)
    if m: break
  return m

QUERY_RE = r"(.*)\((.*)-([\d\']+), (.*)-([\d\']+)\)"
FORBIDDEN_RELS = ["acl:relcl"]
def convert_to_query(query_lines):
  joined_lines = "".join(query_lines)
  q = make_simple_query(joined_lines)
  return q

if __name__ == "__main__":
  queries = []
  query_lines = []
  got_question = False
  question = None
  for line in sys.stdin:
    sline = line.strip()
    if sline == "" and not got_question:
      got_question = True
      question = query_lines[0]
      query_lines = []
    elif sline == "":
      got_question = False
      query = convert_to_query(query_lines)
      queries.append(query)

      print question
      print "\n".join(query_lines)
      if query is None:
        print "none"
      else:
        print query
      print

      query_lines = []
    else:
      query_lines.append(sline)

  #print len(queries)
