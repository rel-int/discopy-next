# -*- coding: utf-8 -*-

"""
ZX simplification benchmarks: :func:`discopy.quantum.graphlike.full_reduce`,
the rules of PyZX's ``full_reduce`` as double-pushout rewriting of
hypergraphs, against PyZX's own on the same random circuits of CNOTs,
Hadamards and T gates.

Both inputs are built before the timer starts, PyZX's graph from its circuit
and ours from that graph; the measured operation copies PyZX's graph, as its
``full_reduce`` rewrites in place, and builds our rewriter from the
hypergraph.
"""

import random
from functools import partial

import pytest

from discopy.quantum import graphlike
from benchmark.config import case, sizes

zx = pytest.importorskip("pyzx")
case = partial(case, suite="zx")


def circuit(qubits, depth):
    """
    PyZX's random circuit of CNOTs, Hadamards and phases, as a graph, from
    seed zero, with the global random state as it was before.
    """
    state = random.getstate()
    random.seed(0)
    try:
        return zx.generate.CNOT_HAD_PHASE_circuit(qubits, depth).to_graph()
    finally:
        random.setstate(state)


@case("graphlike", "10 qubits", sizes(100, 500, full=(1000, )))
def test_graphlike_10(n):
    return partial(graphlike.full_reduce, graphlike.from_pyzx(circuit(10, n)))


@case("pyzx", "10 qubits", sizes(100, 500, full=(1000, )))
def test_pyzx_10(n):
    graph = circuit(10, n)
    return lambda: zx.full_reduce(graph.copy())


@case("graphlike", "20 qubits", sizes(500, full=(1000, )))
def test_graphlike_20(n):
    return partial(graphlike.full_reduce, graphlike.from_pyzx(circuit(20, n)))


@case("pyzx", "20 qubits", sizes(500, full=(1000, )))
def test_pyzx_20(n):
    graph = circuit(20, n)
    return lambda: zx.full_reduce(graph.copy())
