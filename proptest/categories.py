"""
The categories of the property matrix and their parametrisation.

Every file of the suite quantifies over the same list, so it lives here
rather than in any one of them.
"""

import pytest

from discopy import biclosed, cat, closed, ehypergraph, group, markov, matrix
from discopy.utils import factory_name

CATEGORIES = (
    cat.Arrow, biclosed.TermBase, closed.TermBase, matrix.Matrix[int],
    group.Cyclic[4], group.Symmetric[4],
    group.Product[group.Cyclic[2], group.Symmetric[3]],
    group.Wreath[group.Symmetric[3], group.Cyclic[2]],
    ehypergraph.EHypergraph[markov.Diagram])


def category_parameters(classify=lambda category: ()):
    """
    One pytest parameter per category, marked by the given classification,
    a function from a category to its marks, e.g. an expected failure.
    """
    for category in CATEGORIES:
        yield pytest.param(
            category, marks=classify(category), id=factory_name(category))
