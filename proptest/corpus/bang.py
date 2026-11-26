"""
The corpus of :class:`discopy.bang.Diagram`, aimed at what its strategy does
not draw: the structural boxes of a linear category, i.e. dereliction,
digging, contraction, weakening, exchange and laxator.
"""

from discopy.bang import (
    Box, Contraction, Dereliction, Diagram, Digging, Exchange, Laxator, Ty,
    Weakening)

x, y = Ty('x'), Ty('y')
f = Box('f', x, y)
twice = f.bang() >> Contraction(y) >> Dereliction(y) @ Dereliction(y)
structural = twice @ Digging(x) @ Weakening(x) @ Laxator(x, y)\
    @ Exchange(x, y)

CORPUS = {Diagram: {
    "dagger_involution": {
        "dereliction": lambda: {"f": Dereliction(x)},
        "digging": lambda: {"f": Digging(x)},
        "weakening": lambda: {"f": Weakening(x)},
        "laxator": lambda: {"f": Laxator(x, y)},
        "exchange": lambda: {"f": Exchange(x.bang(), y)},
    },
    "dagger_contravariance": {
        "a bang copied and derelicted": lambda: {
            "f": f.bang(), "g": Contraction(y) >> Dereliction(y) @ y.bang()},
    },
    "str_transparency": {
        "every structural box": lambda: {"term": structural},
    },
    "repr_transparency": {
        "every structural box": lambda: {"term": structural},
    },
    "serialisation": {
        "every structural box": lambda: {"term": structural},
    },
}}
