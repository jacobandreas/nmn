#!/usr/bin/env python2

import re
import sys

## # helpers
## 
## def get_with_descendants(node, edges):
##   desc = [get_with_descendants(tail, edges) for head, tail in edges.keys() if
##       head == node]
##   return [node] + sum(desc, [])
## 
## def linearize(nodes):
##   int_nodes = [(int(p.replace("'", "")), (w, p)) for w, p in nodes]
##   srt = sorted(int_nodes)
##   return [s for i, s in srt]
## 
## # transformations
## 
## BORING_DETS = ["the", "a", "an", "this", "some", "any"]
## def remove_boring_det(edges, root_node):
##   new_edges = {ends: rel for ends,rel in edges.items() if ends[1][0] not in
##       BORING_DETS}
##   return new_edges, root_node
## 
## AUX = ["aux", "auxpass", "mark"]
## def remove_aux(edges, root_node):
##   new_edges = {ends: rel for ends, rel in edges.items() if rel not in AUX}
##   return new_edges, root_node
## 
## def remove_case(edges, root_node):
##   new_edges = {ends: rel for ends, rel in edges.items() if rel != "case"}
##   return new_edges, root_node
## 
## def yes_no_question(edges, root_node):
##   maybe_cop = [ends for ends, rel in edges.items() if rel == "cop"]
##   if len(maybe_cop) == 0:
##     return edges, root_node
##   cop = maybe_cop[0]
## 
##   cop_head, cop_tail = cop
##   new_cop = cop_tail, cop_head
##   new_edges = dict(edges)
##   del new_edges[cop]
##   new_edges[new_cop] = "_"
##   new_root = cop_tail
## 
##   return new_edges, new_root
## 
## WH_WORDS = ["who", "what", "where", "when", "why"]
## def wh_question(edges, root_node):
##   maybe_subj_edge = [ends for ends, rel in edges.items() if rel in ("nsubj", "dep", "advmod")]
## 
##   for subj_edge in maybe_subj_edge:
##     vrb, subj = subj_edge
##     below_subj = get_with_descendants(subj, edges)
##     below_subj_lin = linearize(below_subj)
##     if not any(w[0] in WH_WORDS for w in below_subj_lin):
##       continue
## 
##     objs = [tail for head, tail in edges.keys() if head == vrb and tail != subj]
## 
##     collapsed_label = "_".join([w[0] for w in below_subj_lin])
##     collapsed = (collapsed_label, subj[1])
## 
##     new_edges = dict(edges)
##     del new_edges[subj_edge]
##     new_root = collapsed
## 
##     for obj in objs:
##       del new_edges[vrb,obj]
##       new_edges[(collapsed, obj)] = "_"
## 
##     return new_edges, new_root
## 
##   return edges, root_node
## 
## def inject_nmod(edges, root_node):
##   new_edges = dict(edges)
##   while True:
##     nmod_edges = [(ends, rel) for ends, rel in new_edges.items() if "nmod:" in rel]
##     if len(nmod_edges) == 0:
##       break
##     nmod_edge, rel = nmod_edges[0]
##     nmod_edges = nmod_edges[1:]
## 
##     head, tail = nmod_edge
##     mod = rel.split(":")[1]
##     head_in_edges = [(h, t) for h, t in new_edges.keys() if t == head]
## 
##     new_parent = ("and", "_" + head[1])
##     new_mod = (mod, "_" + tail[1])
##     if nmod_edge in new_edges: del new_edges[nmod_edge]
##     new_edges[new_parent, head] = "_"
##     new_edges[new_parent, new_mod] = "_"
##     new_edges[new_mod, tail] = "_"
## 
##     for head_in_edge in head_in_edges:
##       del new_edges[head_in_edge]
##       new_edges[head_in_edge[0],new_parent] = "_"
## 
##   return new_edges, root_node
## 
## QUERY_TRANSFORMATIONS = [
##   remove_boring_det, 
##   remove_aux, 
##   remove_case,
##   yes_no_question,
##   wh_question
##   #inject_nmod
## ]
## def stringify(edges, root_node, used_roots = None):
##   if used_roots is None: used_roots = set()
##   if root_node in used_roots: return "LOOP"
##   used_roots.add(root_node)
##   children = [tail for head, tail in edges.keys() if head == root_node]
##   if len(children) == 0:
##     return root_node[0]
##   return "(%s %s)" % (root_node[0], " ".join([stringify(edges, c, used_roots) for c in children]))


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
  #edges = dict()
  #for line in query_lines:
  #  m = re.match(QUERY_RE, line)
  #  rel = m.group(1)
  #  if rel in FORBIDDEN_RELS:
  #    continue
  #  head = (m.group(2).lower(), m.group(3))
  #  tail = (m.group(4).lower(), m.group(5))
  #  assert (head,tail) not in edges
  #  edges[head,tail] = rel

  joined_lines = "".join(query_lines)
  q = make_simple_query(joined_lines)
  return q

  #root_node = [tail for head, tail in edges.keys() if head == ("root", "0")][0]
  #for transformation in QUERY_TRANSFORMATIONS:
  #  edges, root_node = transformation(edges, root_node)

  #q = stringify(edges, root_node)

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

      #print question
      #print "\n".join(query_lines)
      if query is None:
        print "none"
      else:
        print query
      #print

      query_lines = []
    else:
      query_lines.append(sline)

  #print len(queries)
