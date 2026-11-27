"""
The corpus of :mod:`discopy.quantum.zx`: green and red spiders with dyadic,
non-dyadic and decimal float phases, the Hadamard :data:`~discopy.quantum.zx.H`
and Clifford scalars.
"""

from fractions import Fraction

from discopy.quantum.zx import Diagram, H, Hypergraph, Scalar, X, Z

composite = (
    Z(1, 2, Fraction(1, 3))
    >> X(2, 1, 0.1)
    >> H @ Scalar(2j) @ Scalar(2 ** 0.5)
)

CORPUS = {
    Diagram: {
        "pickling": {
            "a Hadamard": lambda: {"term": H >> X(1, 1)},
            "a third-turn spider and a root-two scalar": lambda: {
                "term": Z(1, 1, Fraction(1, 3)) @ Scalar(2 ** 0.5)},
        },
        "repr_transparency": {
            "dyadic and third-turn spiders, Hadamard and scalars": lambda: {
                "term": Z(1, 1, Fraction(1, 4)) >> composite},
        },
        "str_transparency": {
            "third-turn spiders, Hadamard and scalars": lambda: {
                "term": composite},
        },
        "hypergraph_composition": {
            "decimal float phases": lambda: {
                "f": Z(1, 1, 0.1), "g": Z(1, 1, 0.2)},
        },
        "hypergraph_section": {
            "spiders, Hadamard and Clifford scalars": lambda: {
                "f": composite},
        },
    },
    Hypergraph: {
        "pickling": {
            "a red spider": lambda: {"term": Hypergraph.from_diagram(X(1, 2))},
        },
        "repr_transparency": {
            "spiders, Hadamard and Clifford scalar": lambda: {
                "term": Hypergraph.from_diagram(composite)},
        },
        "str_transparency": {
            "a third-turn green spider": lambda: {
                "term": Hypergraph.from_diagram(Z(1, 1, Fraction(1, 3)))},
            "a red spider, Hadamard and root-two scalar": lambda: {
                "term": Hypergraph.from_diagram(composite)},
        },
        "dagger_involution": {
            "a decimal float phase": lambda: {
                "f": Hypergraph.from_diagram(Z(1, 1, 0.1))},
            "a red spider and Clifford scalar": lambda: {
                "f": Hypergraph.from_diagram(
                    X(1, 2, Fraction(1, 3)) @ Scalar(2j))},
        },
        "hypergraph_roundtrip": {
            "an imaginary Clifford scalar": lambda: {
                "graph": Hypergraph.from_diagram(Scalar(2j))},
            "a root-two scalar": lambda: {
                "graph": Hypergraph.from_diagram(Scalar(2 ** 0.5))},
            "spiders, Hadamard and Clifford scalars": lambda: {
                "graph": Hypergraph.from_diagram(composite)},
        },
    },
}
