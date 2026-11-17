# -*- coding: utf-8 -*-

"""
E-graph benchmarks: the traces of ``benchmark/data/egraph-traces``, i.e. the
e-graphs a program search grew on 50 tasks of 1D-ARC, replayed through
:class:`discopy.egraph.Builder`, through the pure
:class:`discopy.egraph.EGraph` on the small ones, and through the
bookkeeping of the search itself, a dictionary and two lists, which the
builder is measured against on one machine. The family ``freeze`` times the
:class:`discopy.egraph.EGraph` a builder gives back.

The family ``match`` times e-matching on each trace, of the two-box pattern
with the most matches there, by :meth:`discopy.egraph.Builder.matches` on
the builder and by :meth:`discopy.egraph.EGraph.matches` on what it freezes
into. The family ``saturate`` times :meth:`discopy.egraph.EGraph.saturate`
under associativity and commutativity of ``+`` on ``n`` leaves, which
saturates to ``2 ** n - 1`` classes, against the pure rounds of
:meth:`discopy.egraph.EGraph.step`, its size ``n``.

A case is how its traces were chosen, the 25 with the most e-nodes or 25 of
the others at random, and its size ``n`` the rank of a trace among them, in
order of e-nodes: the README of the data says how many each has.

No trace ever relates two classes that are both read, so the family
``rebuild`` replays a trace twice on separate base classes, relates them and
rebuilds, which relates every class of one copy to one of the other.
"""

import gzip
import json
from collections import Counter, defaultdict
from functools import cache, partial, reduce
from pathlib import Path
from typing import NamedTuple

from discopy.egraph import Builder, EGraph, Markov
from discopy.hypergraph import Hypergraph
from discopy.markov import Box, Diagram, Swap, Ty
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


def most_matched(trace: Trace) -> Hypergraph:
    """ The pattern of a box read by another at some input with the most
    matches on a trace, counted from its rows: each reading of a class by
    the second, once for each e-node of the first in that class. """
    producers: defaultdict[int, Counter] = defaultdict(Counter)
    for head, out, *_ in trace.rows:
        producers[out][head] += 1
    counts: Counter = Counter()
    for head, _, *args in trace.rows:
        for k, arg in enumerate(args):
            for first, number in producers[arg].items():
                counts[first, head, k] += number
    (first, second, k), _ = counts.most_common(1)[0]
    reader = trace.boxes[second]
    return (X ** k @ trace.boxes[first] @ X ** (len(reader.dom) - 1 - k)
            >> reader).to_hypergraph()


def matching(chosen: str, n: int, pure: bool = False):
    """ The matches of :func:`most_matched` on a built trace, on the builder
    or on what it freezes into, its relations or producers indexed once. """
    trace = load(chosen, n)
    graph, lhs = build(trace), most_matched(trace)
    graph = graph.freeze() if pure else graph
    graph.matches(lhs)
    return partial(graph.matches, lhs)


PLUS = Box('+', X @ X, X)
AC = ((PLUS, Swap(X, X) >> PLUS), (PLUS @ X >> PLUS, X @ PLUS >> PLUS))


def leaves(n: int) -> EGraph:
    """ The sum of ``n`` leaves, bracketed to the left. """
    return Markov.from_diagram(reduce(Diagram.then, (
        PLUS @ X ** k for k in reversed(range(n - 1))), Diagram.id(X ** n)))


def saturate(n: int) -> EGraph:
    """ The last round of saturating the sum of ``n`` leaves. """
    graph = leaves(n)
    *_, last = graph, *graph.saturate(*AC)
    return last


def pure(n: int) -> EGraph:
    """ The rounds of :meth:`EGraph.step` on the sum of ``n`` leaves until
    one adds no box and relates no class, as :meth:`EGraph.saturate` did
    before it ran on the builder. """
    rules = [tuple(side.to_hypergraph() for side in rule) for rule in AC]
    graph = leaves(n)
    while True:
        new = graph.step(rules)
        if len(new.graph.boxes) == len(graph.graph.boxes) and len(set(
                new.classes)) == len(set(graph.classes)):
            return new
        graph = new


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


@case("match", "builder, largest", sizes(*range(1, 26)))
def test_match_builder_largest(n):
    return matching("largest", n)


@case("match", "pure, largest", sizes(*range(1, 26)))
def test_match_pure_largest(n):
    return matching("largest", n, pure=True)


@case("match", "builder, random", sizes(*range(1, 26)))
def test_match_builder_random(n):
    return matching("random", n)


@case("match", "pure, random", sizes(*range(1, 26)))
def test_match_pure_random(n):
    return matching("random", n, pure=True)


@case("saturate", "builder", sizes(4, 5, 6, 7, full=(8, )))
def test_saturate_builder(n):
    return partial(saturate, n)


@case("saturate", "pure", sizes(4, 5))
def test_saturate_pure(n):
    return partial(pure, n)
