"""
The corpus of :class:`discopy.hypergraph.Hypergraph` over a closed category,
aimed at what its strategy does not draw: the hypergraph of a curry, which
has no dagger, see ``closed.Diagram.dagger_involution``.
"""

from discopy import closed
from discopy.lambek import abstract


def curried(module):
    """ The hypergraph of a curry in a module, and of an identity beside. """
    x, y = module.Ty('x'), module.Ty('y')
    curry = module.Box('f', x @ y, x).curry().to_hypergraph()
    return curry, module.Diagram.id(x).to_hypergraph()


CORPUS = {
    module.Hypergraph: {
        "dagger_involution": {
            "a curry": lambda module=module: {"f": curried(module)[0]},
        },
        "dagger_contravariance": {
            "an identity then a curry": lambda module=module: {
                "f": curried(module)[1], "g": curried(module)[0]},
        },
        "dagger_monoidality": {
            "a curry beside an identity": lambda module=module: {
                "f": curried(module)[0], "g": curried(module)[1]},
        },
    } for module in (closed, abstract)}
