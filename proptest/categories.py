"""
The categories of the property matrix and their parametrisation.

Every file of the suite quantifies over the same list, so it lives here
rather than in any one of them.
"""

import pytest

from discopy import (
    balanced, biclosed, braided, cat, closed, compact, feedback, frobenius,
    markov, monoidal, pivotal, ribbon, rigid, symmetric, traced)
from discopy.cmap import CMap
from discopy.hypergraph import Hypergraph
from discopy.utils import factory_name, get_origin

DIAGRAMS = tuple(module.Diagram for module in (
    monoidal, braided, balanced, traced, biclosed, rigid, pivotal, ribbon,
    symmetric, compact, markov, feedback, closed, frobenius))
""" The free categories of the hierarchy, from planar to hypergraph. """

QUOTIENTS = (
    CMap[symmetric.Diagram], CMap[compact.Diagram],
    Hypergraph[markov.Diagram], Hypergraph[frobenius.Diagram])
""" The combinatorial structures the symmetric diagrams are compared up to.
"""

CATEGORIES = (cat.Arrow, ) + DIAGRAMS + QUOTIENTS


def category_id(category) -> str:
    """ The name of a category, with the parameter of a quotient. """
    parameter = getattr(category, "category", None)
    if parameter is None or parameter is category:
        return factory_name(category)
    return f"{factory_name(get_origin(category))}[{factory_name(parameter)}]"


def category_parameters(classify=lambda category: ()):
    """
    One pytest parameter per category, marked by the given classification,
    a function from a category to its marks, e.g. an expected failure.
    """
    for category in CATEGORIES:
        yield pytest.param(
            category, marks=classify(category), id=category_id(category))
