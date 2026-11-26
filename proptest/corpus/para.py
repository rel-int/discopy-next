"""
The corpus of :class:`discopy.para.Symmetric` and its extensions, aimed at what
their strategy does not draw: parameters that differ on both sides of a law,
equal parameters and coparameters, adjoint parameters, a memory and a trace
with a parameter.
"""

from discopy import compact, frobenius, markov
from discopy.para import Compact, Feedback, Hypergraph, Symmetric, Traced
from discopy.symmetric import Box, Ty

x, y, z, p, q, p1, q1, a = map(Ty, "x y z p q p1 q1 a".split())
e, ca = Ty(), compact.Ty('a')
f = Symmetric(x, y, Box('f', x @ p, y), p)
g = Symmetric(y, z, Box('g', y @ q, z), q)
h = Symmetric(z, x, Box('h', z @ p1, x), p1)
k = Symmetric(x, y, Box('k', x @ q1, y), q1)
scalar = Symmetric(e, e, Box('s', q1, e), q1)
t = Traced(x @ y, x @ z, Box('t', x @ y @ p, x @ z), p)
u = Traced(x, x, Box('u', x @ q, x), q)
t_right = Traced(y @ x, z @ x, Box('t', y @ x @ p, z @ x), p)
dinaturality = {"M": x, "N": x, "S": e, "T": e, "Γ": y, "Δ": z}


def swapped(terms):
    """ The terms of ``f, g, h, k`` as ``f, h, g, k``. """
    return terms[0], terms[2], terms[1], terms[3]


def named(terms):
    """ The arguments ``f, g, h, k`` of a law, given its four terms. """
    return dict(zip("fghk", terms))


def coparameters():
    """ Four maps ``x -> y -> z -> x -> y`` with equal param and copar. """
    return [Symmetric(dom, cod, Box(name, dom @ a, cod @ a), a, a)
            for name, dom, cod in (
                ('f', x, y), ('g', y, z), ('h', z, x), ('k', x, y))]


def adjoints(params):
    """ Four maps on one wire with the parameters ``params`` of a compact. """
    return [Compact(ca, ca, compact.Box(name, ca @ params, ca @ ca),
                    params, ca) for name in "fghk"]


def hypergraphs():
    """ Four hypergraph maps with equal param and copar on one wire. """
    b = frobenius.Ty('a')
    boxes = [frobenius.Box(name, b @ b, b @ b @ b if name == 'f' else b @ b)
             for name in "fghk"]
    return [Hypergraph(b, b, box, b, box.cod[1:]) for box in boxes]


def memory():
    """ Four feedback maps ``X -> Y -> Z -> X -> Y`` with a memory ``M``. """
    X, Y, Z, M = map(markov.Ty, "XYZM")
    return [Feedback(dom, cod, markov.Box(name, dom @ M, cod @ M), M, M)
            for name, dom, cod in (
                ('f', X, Y), ('g', Y, Z), ('h', Z, X), ('k', X, Y))]


def mixed():
    """ Four feedback maps on one wire, with a param, a copar or both. """
    X, A = map(markov.Ty, "XA")
    return [
        Feedback(X, X, markov.Box('f', X, X)),
        Feedback(X, X, markov.Box('g', X @ A, X @ A), A, A),
        Feedback(X, X, markov.Box('h', X @ A, X), A),
        Feedback(X, X, markov.Box('k', X, X @ A), copar=A)]


def transposable():
    """ Two composable compact maps, each with a parameter. """
    A, B, C, P, Q = map(compact.Ty, "ABCPQ")
    return {"f": Compact(A, B, compact.Box('f', A @ P, B), P),
            "g": Compact(B, C, compact.Box('g', B @ Q, C), Q)}


three = ca @ ca.r @ ca
seven = ca @ ca.r @ ca @ ca.r @ ca @ ca @ ca

CORPUS = {
    Symmetric: {
        "bifunctoriality": {
            "four parameters": lambda: named(swapped([f, g, h, k])),
            "equal param and copar": lambda: named(
                swapped(coparameters())),
        },
        "braid_naturality": {
            "two parameters": lambda: {"f": f, "g": g},
            "equal param and copar": lambda: dict(
                zip("fg", coparameters())),
        },
        "scalar_sliding": {
            "a scalar with a parameter": lambda: {"scalar": scalar, "f": f},
        },
    },
    Traced: {
        "trace_dinaturality_left": {
            "a parameter on both": lambda: {"f": t, "g": u, **dinaturality},
        },
        "trace_dinaturality_right": {
            "a parameter on both": lambda: {
                "f": t_right, "g": u, **dinaturality},
        },
    },
    Compact: {
        "bifunctoriality": {
            "three adjoint wires": lambda: named(adjoints(three)),
            "three adjoint wires, crossed": lambda: named(
                swapped(adjoints(three))),
            "seven adjoint wires": lambda: named(adjoints(seven)),
            "seven adjoint wires, crossed": lambda: named(
                swapped(adjoints(seven))),
        },
        "transpose_contravariance": {
            "parameters on both": transposable,
        },
    },
    Feedback: {
        "bifunctoriality": {
            "a memory": lambda: named(swapped(memory())),
            "a param, a copar and both": lambda: named(mixed()),
        },
        "braid_naturality": {
            "a memory": lambda: dict(zip("fg", memory())),
        },
    },
    Hypergraph: {
        "bifunctoriality": {
            "equal param and copar": lambda: named(hypergraphs()),
            "equal param and copar, crossed": lambda: named(
                swapped(hypergraphs())),
        },
        "braid_naturality": {
            "equal param and copar": lambda: dict(zip("fg", hypergraphs())),
        },
    },
}
