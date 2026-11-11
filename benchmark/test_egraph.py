# -*- coding: utf-8 -*-

"""
E-graph benchmarks: the traces of ``benchmark/data/egraph-traces``, i.e. the
e-graphs a program search grew on 50 tasks of 1D-ARC, replayed through
:class:`discopy.egraph.Builder`, through the pure
:class:`discopy.egraph.EGraph` on the small ones, and through the
bookkeeping of the search itself, a dictionary and two lists, which the
builder is measured against on one machine. The family ``freeze`` times the
:class:`discopy.egraph.EGraph` a builder gives back.

A case is how its traces were chosen, the 25 with the most e-nodes or 25 of
the others at random, and its size ``n`` the rank of a trace among them, in
order of e-nodes: the README of the data says how many each has.

No trace ever relates two classes that are both read, so the family
``rebuild`` replays a trace twice on separate base classes, relates them and
rebuilds, which relates every class of one copy to one of the other.
"""

import gzip
import json
from functools import cache, partial
from pathlib import Path
from typing import NamedTuple

from discopy.egraph import Builder, EGraph, Markov
from discopy.markov import Box, Ty
from benchmark.config import case, sizes

case = partial(case, suite="egraph")

TRACES = Path(__file__).parent / "data" / "egraph-traces"
X, EMPTY = Ty('x'), Markov.id(Ty())


class Trace(NamedTuple):
    """ The rows of a trace, a box for each of its heads, those of a second
    copy with base classes of its own, and how many base classes it has. """
    rows: list
    boxes: list
    others: list
    bases: int


@cache
def load(chosen: str, n: int) -> Trace:
    """ The ``n``-th trace chosen some way, from one. """
    index = json.loads((TRACES / "index.json").read_text())["traces"]
    name = [item["file"] for item in index if item["chosen"] == chosen][n - 1]
    with gzip.open(TRACES / name, "rt") as file:
        trace = json.load(file)
    boxes, others = ([
        Box(head["name"] + suffix * (not head["arity"]),
            Ty().tensor(*head["arity"] * [X]), X)
        for head in trace["heads"]] for suffix in ("", "'"))
    return Trace(trace["rows"], boxes, others, sum(
        not head["arity"] for head in trace["heads"]))


def search(trace: Trace) -> list:
    """ The bookkeeping of the search on a trace, as its README gives it. """
    seen, enodes, kinds, out = {}, [], [], []
    for row in trace.rows:
        key = (row[0], tuple(row[2:]))
        cid = seen.get(key)
        if cid is None:
            cid = row[1]
            if cid == len(kinds):
                kinds.append(row[0])
            seen[key] = cid
            enodes.append(key + (cid,))
        out.append(cid)
    return out


def replay(builder: Builder, boxes: list, rows: list) -> list:
    """ The e-nodes of a trace added to a builder in turn, each related to
    the class recorded for it: the spider standing for each class. """
    add, union, classes = builder.add, builder.union, []
    for head, out, *args in rows:
        spider, = add(boxes[head], [classes[i] for i in args])
        if out == len(classes):
            classes.append(spider)
        else:
            union(spider, classes[out])
    return classes


def build(trace: Trace) -> Builder:
    """ A trace through a builder. """
    builder = EMPTY.builder()
    replay(builder, trace.boxes, trace.rows)
    return builder


def rebuild(trace: Trace) -> Builder:
    """ Two copies of a trace through a builder, their base classes related,
    then rebuilt. """
    builder = EMPTY.builder()
    classes, others = (
        replay(builder, boxes, trace.rows)
        for boxes in (trace.boxes, trace.others))
    for pair in zip(classes[:trace.bases], others):
        builder.union(*pair)
    builder.rebuild()
    return builder


def fold(trace: Trace) -> EGraph:
    """ A trace through the pure e-graph. """
    graph, classes = EMPTY, []
    for head, out, *args in trace.rows:
        graph, (spider, ) = graph.add(
            trace.boxes[head], tuple(classes[i] for i in args))
        if out == len(classes):
            classes.append(spider)
        else:
            graph = graph.merge(spider, classes[out])
    return graph


@case("search", "random", sizes(*range(1, 26)))
def test_search_random(n):
    return partial(search, load("random", n))


@case("search", "largest", sizes(*range(1, 26)))
def test_search_largest(n):
    return partial(search, load("largest", n))


@case("builder", "random", sizes(*range(1, 26)))
def test_builder_random(n):
    return partial(build, load("random", n))


@case("builder", "largest", sizes(*range(1, 26)))
def test_builder_largest(n):
    return partial(build, load("largest", n))


@case("rebuild", "random", sizes(*range(1, 26)))
def test_rebuild_random(n):
    return partial(rebuild, load("random", n))


@case("rebuild", "largest", sizes(*range(1, 26)))
def test_rebuild_largest(n):
    return partial(rebuild, load("largest", n))


@case("freeze", "largest", sizes(1, 25))
def test_freeze_largest(n):
    return build(load("largest", n)).freeze


@case("pure", "random", sizes(1, 3, 12, 15, 23, full=(24, 25)))
def test_pure_random(n):
    return partial(fold, load("random", n))
