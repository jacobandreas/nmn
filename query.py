#!/usr/bin/env python2

import sexpdata

def parse_query(query):
    parsed = sexpdata.loads(query)
    extracted = extract_query(parsed)
    return extracted

def extract_query(sexp_query):
    if isinstance(sexp_query, sexpdata.Symbol):
        return sexp_query.value()
    elif isinstance(sexp_query, int):
        return str(sexp_query)
    elif isinstance(sexp_query, bool):
        return str(sexp_query).lower()
    return tuple(extract_query(q) for q in sexp_query)
