"""
The corpus of :class:`discopy.symmetric.Diagram`, aimed at what its strategy
never draws: boxes that share a name, carry ``data`` or ``is_dagger``, names
that are not identifiers, bubbles, empty and long types, scalars, traces of
several wires and feedback with a memory of several wires or delays.
"""

from discopy.symmetric import Box, Diagram, Swap, Ty

(x, y, z, m, n), e = map(Ty, "xyzmn"), Ty()
f, g = Box("f", x, y), Box("f", y @ y, x)
odd = Box("f(x) >> g @ 'h'", x @ y, e)
dagger = Box("d", x, y @ z, is_dagger=True)
data = Box("w", x, x, data={"a": [1, 2.5, {"b": None}]})
state, effect, scalar = Box("s", e, x @ y), Box("t", z, e), Box("c", e, e)
long = Ty(*"abcdefghij")
permutation = Diagram.permutation((9, 3, 0, 7, 1, 8, 2, 6, 4, 5), long)
bubble = (f >> Box("g", y, x)).bubble(dom=x, cod=x, name="B")
k = Box("k", x @ y @ z, x @ y @ z)
nested = k.trace(left=True).trace(left=True)
delayed = Diagram.id(m @ n).delay().dom
memory = Box("f", x @ delayed, y @ m @ n)
loop = memory.feedback(dom=x, cod=y, mem=m @ n)

CORPUS = {Diagram: {
    "repr_transparency": {
        "same name, other type": lambda: {"term": f @ g},
        "odd name": lambda: {"term": odd},
        "dagger flag": lambda: {"term": dagger >> dagger.dagger()},
        "data": lambda: {"term": data >> data},
        "bubble": lambda: {"term": bubble @ bubble.bubble(name="C")},
        "nested feedback": lambda: {"term": loop.delay()},
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
    "hexagon_left": {
        "empty in the middle": lambda: {"x": x @ y, "y": e, "z": long},
    },
    "yanking": {
        "an empty type": lambda: {"x": e},
    },
    "hypergraph_composition": {
        "same name, other type": lambda: {"f": f, "g": Box("f", y, x)},
        "nested traces": lambda: {"f": nested, "g": nested},
    },
    "hypergraph_section": {
        "bubble": lambda: {"f": bubble},
        "data": lambda: {"f": data},
        "a trace of two wires, right": lambda: {"f": k.trace(2)},
    },
    "foliation_idempotence": {
        "scalars between swaps": lambda: {
            "f": Swap(x, y) >> scalar @ y @ x >> Swap(y, x) @ scalar},
        "nested traces": lambda: {"f": nested @ nested},
    },
    "joining": {
        "a box under the delays": lambda: {
            "f": Box("f", x @ y @ delayed, z @ m @ n),
            "x": x @ y, "y": z, "m": m, "n": n},
        "an empty memory": lambda: {
            "f": Box("f", x @ Diagram.id(m).delay().dom, y @ m),
            "x": x, "y": y, "m": m, "n": e},
    },
}}
