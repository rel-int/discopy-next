"""
The corpus of :class:`discopy.symmetric.Diagram`, aimed at what its strategy
draws rarely or never: boxes that share a name, carry ``data`` or
``is_dagger``, names that are not identifiers, bubbles and long types, and of
:class:`discopy.feedback.symmetric.Diagram`: traces of several wires and
feedback with a memory of several wires.
"""

from discopy import symmetric
from discopy.feedback import symmetric as traced
from discopy.symmetric import Box, Diagram, Swap, Ty

(x, y, z), e = map(Ty, "xyz"), Ty()
f, g = Box("f", x, y), Box("f", y @ y, x)
odd = Box("f(x) >> g @ 'h'", x @ y, e)
dagger = Box("d", x, y @ z, is_dagger=True)
data = Box("w", x, x, data={"a": [1, 2.5, {"b": None}]})
state, effect, scalar = Box("s", e, x @ y), Box("t", z, e), Box("c", e, e)
long = Ty(*"abcdefghij")
permutation = Diagram.permutation((9, 3, 0, 7, 1, 8, 2, 6, 4, 5), long)
bubble = (f >> Box("g", y, x)).bubble(dom=x, cod=x, name="B")

(tx, ty, tz, m, n), te = map(traced.Ty, "xyzmn"), traced.Ty()
k = traced.Box("k", tx @ ty @ tz, tx @ ty @ tz)
nested = k.trace(left=True).trace(left=True)
delayed = traced.Diagram.id(m @ n).delay().dom
memory = traced.Box("f", tx @ delayed, ty @ m @ n)
loop = memory.feedback(dom=tx, cod=ty, mem=m @ n)

CORPUS = {symmetric.Diagram: {
    "repr_transparency": {
        "same name, other type": lambda: {"term": f @ g},
        "odd name": lambda: {"term": odd},
        "dagger flag": lambda: {"term": dagger >> dagger.dagger()},
        "data": lambda: {"term": data >> data},
        "bubble": lambda: {"term": bubble @ bubble.bubble(name="C")},
    },
    "serialisation": {
        "data and dagger": lambda: {"term": data @ dagger},
        "bubble": lambda: {"term": bubble},
        "odd name": lambda: {"term": odd},
    },
    "associativity": {
        "states, effects and a scalar": lambda: {
            "f": state, "g": x @ Box("h", y, z), "h": x @ effect @ scalar},
    },
    "braid_naturality": {
        "same name, other type": lambda: {"f": f, "g": g},
        "a state and ten wires": lambda: {"f": state, "g": permutation},
    },
    "hypergraph_composition": {
        "same name, other type": lambda: {"f": f, "g": Box("f", y, x)},
    },
    "hypergraph_section": {
        "bubble": lambda: {"f": bubble},
        "data": lambda: {"f": data},
    },
    "foliation_idempotence": {
        "scalars between swaps": lambda: {
            "f": Swap(x, y) >> scalar @ y @ x >> Swap(y, x) @ scalar},
    },
}, traced.Diagram: {
    "repr_transparency": {
        "nested feedback": lambda: {"term": loop.delay()},
    },
    "hypergraph_composition": {
        "nested traces": lambda: {"f": nested, "g": nested},
    },
    "hypergraph_section": {
        "a trace of two wires, right": lambda: {"f": k.trace(2)},
    },
    "foliation_idempotence": {
        "nested traces": lambda: {"f": nested @ nested},
    },
    "joining": {
        "a box under the delays": lambda: {
            "f": traced.Box("f", tx @ ty @ delayed, tz @ m @ n),
            "x": tx @ ty, "y": tz, "m": m, "n": n},
        "an empty memory": lambda: {
            "f": traced.Box("f", tx @ traced.Diagram.id(m).delay().dom,
                            ty @ m),
            "x": tx, "y": ty, "m": m, "n": te},
    },
}}
