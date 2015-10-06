#!/usr/bin/env python2

import re
import sys

def get_chunks(words):
    buf = []
    current_chunk = None
    current_words = []
    for ann_word in words:
        word, tag, chunk = ann_word.split("|")
        if tag in ("DT", "RB"):
            continue
        if chunk == "O":
            chunk_pos = "O"
            chunk_type = None
        else:
            chunk_pos, chunk_type = chunk.split("-")
        if chunk_pos == "B" or chunk_type != current_chunk:
            buf.append((current_chunk, current_words))
            current_words = []
            current_chunk = chunk_type

        #if word[-1] == "s":
        #    word = word[:-1]
        current_words.append(word)
    buf.append((current_chunk, current_words))
    return buf[1:]

def fuse_chunks(chunks):
    buf = []
    current_type = None
    current_words = []
    for chunk_type, words in chunks:
        if chunk_type == "VP":
            continue
            word = words[-1]
            if word in ("is", "are", "have", "do", "be", "can", "will", "did",
                        "were", "was", "had"):
                continue
            print word
            buf.append((current_type, [word]))
        elif chunk_type == "PP" and words[0] != "of":
            buf.append((current_type, current_words))
            current_type = "PP"
            current_words = [] + words
        else: # chunk_type == "NP":
            current_words += words
    buf.append((current_type, current_words))

    final_buf = []
    for chunk_type, words in buf:
        while "and" in words and "between" not in words:
            pre_and = words[:words.index("and")]
            words = words[words.index("and")+1:]
            if len(pre_and) > 0:
                final_buf.append((chunk_type, pre_and))
        final_buf.append((chunk_type, words))
    return final_buf

FILTER = ["side"]
MAPPINGS = [
    ("on left of", "left_of"),
    ("to left of", "left_of"),
    ("at left of", "left_of"),
    ("left of", "left_of"),
    ("left to", "left_of"),

    ("on right of", "right_of"),
    ("to right of", "right_of"),
    ("at right of", "right_of"),
    ("right of", "right_of"),

    ("in front of", "front_of"),
    ("to front of", "front_of"),
    ("at front of", "front_of"),

    ("in back of", "back_of"),
    ("to back of", "back_of"),
    ("at back of", "back_of"),
    
    ("on top of", "top_of"),
    ("on bottom of", "bottom_of"),

    ("close to", "close_to"),

    ("on", "on"),
    ("in", "in"),
    ("at", "at"),
    ("by", "by"),
    ("above", "above"),
    ("below", "below"),
    ("behind", "behind"),
    ("under", "under"),
    ("around", "around"),
    ("with", "with"),
    ("near", "near"),
]
MAPPED = [m[1] for m in MAPPINGS]

def proc_head(chunk):
    if chunk[:2] == ["how", "many"]:
        return ("count", proc_body(chunk[2:]))
    if chunk[:2] in (["what", "color"], ["what", "colour"]):
        body = chunk[3:] if (len(chunk) > 2 and chunk[2] == "of") else chunk[2:]
        return ("color", proc_body(body))
    elif len(chunk) == 0:
        return ("what", "object")
    elif chunk[0] not in ("what", "where", "color", "count"):
        return ("what", "object")
    else:
        return (chunk[0], proc_body(chunk[1:]))

def proc_body(chunk):
    if len(chunk) == 0:
        return None
    filtered = [c for c in chunk if c not in FILTER]
    string = " ".join(filtered)
    joined = string
    for before, after in MAPPINGS:
        joined = joined.replace(before, after)
    joined = joined.split()
    if len(joined) == 0:
        return None
    if joined[0] == "between":
        return None
    if len(joined) == 1 and joined[0] == "to":
        return None
    if len(joined) == 1 or joined[0] not in MAPPED:
        if not re.match(r"\w+", joined[0]):
            return None
        else:
            return joined[0]
    else:
        if not (re.match(r"\w+", joined[0]) and re.match(r"\w+", joined[-1])):
            return None
        else:
            return (joined[0], joined[-1])

def parse(line):
    print
    print line
    words = line.split()
    chunks = get_chunks(words)
    fchunks = fuse_chunks(chunks)
    question, preds1 = proc_head(fchunks[0][1])
    preds2 = [proc_body(c[1]) for c in fchunks[1:-1] if c[1] != []]
    print fchunks[1:-1]
    print preds2

    parse = [question]
    if preds1 is not None:
        parse.append(preds1)
    for pred in preds2:
        if pred is not None:
            parse.append(pred)
        
    #print "\n".join([str(p) for p in preds2])
    #print question
    #print preds1
    #print preds2

    if len(parse) == 1:
        parse = (parse[0], "object")

    assert len(parse) >= 2
    if len(parse) > 2:
        #parse = (parse[0], ("and",) + tuple(parse[1:]))
        parse = (parse[0], parse[1])

    return tuple(parse)

def flatten(parse):
    if isinstance(parse, tuple):
        return "(%s)" % " ".join([flatten(p) for p in parse])
    else:
        return parse

if __name__ == "__main__":
    for line in sys.stdin:
        #print " ".join([w.split("|")[0] for w in line.strip().split()])
        #print parse(line.strip())
        #print
        print flatten(parse(line.strip()))

