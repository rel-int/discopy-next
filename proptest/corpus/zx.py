"""
The corpus of :mod:`discopy.quantum.zx`, aimed at what its strategies do not
draw: the Hadamard :data:`~discopy.quantum.zx.H`, which every hypergraph of a
red spider holds.
"""

from discopy.quantum.zx import Diagram, H, Hypergraph, X

CORPUS = {
    Diagram: {
        "pickling": {"a Hadamard": lambda: {"term": H >> X(1, 1)}},
    },
    Hypergraph: {
        "pickling": {
            "a red spider": lambda: {"term": Hypergraph.from_diagram(X(1, 2))},
        },
    },
}
