"""
Property tests for the strategies of diagrams and of their quotients.

Every diagram category generates its arrows from the structure up to which
it compares them: a :class:`discopy.cmap.CMap` for symmetric and compact
categories, a :class:`discopy.hypergraph.Hypergraph` for Markov and
hypergraph categories, and the :meth:`discopy.monoidal.Diagram.normal_form`
of a planar diagram otherwise. These tests check that each strategy meets
its contract, that the boundaries it is asked for are honoured and the
structure it claims is there, and that a generated quotient reads back to
itself through the diagram it downgrades to.
"""

import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from discopy import (
    balanced, biclosed, braided, closed, compact, feedback, frobenius, markov,
    monoidal, pivotal, ribbon, rigid, symmetric, traced)
from discopy.cmap import CMap
from discopy.hypergraph import Hypergraph
from discopy.utils import factory_name

PLANAR = (monoidal, braided, balanced, traced, biclosed, rigid, pivotal,
          ribbon)
""" The modules whose diagrams have no symmetry, compared syntactically. """

SYMMETRIC = (symmetric, compact)
""" The modules whose diagrams are generated through a combinatorial map. """

MARKOV = (markov, closed, frobenius)
""" The modules whose diagrams are generated through a hypergraph. """

MODULES = PLANAR + SYMMETRIC + MARKOV + (feedback, )


def parameters(modules):
    """ One pytest parameter per module, named after its diagrams. """
    return [pytest.param(module, id=factory_name(module.Diagram))
            for module in modules]


SPIDERS = {
    "any": lambda graph: True,
    "left_monogamous": lambda graph: graph.is_left_monogamous,
    "monogamous": lambda graph: graph.is_monogamous,
    "bijective": lambda graph: graph.is_bijective,
}
""" What each constraint of :meth:`Hypergraph.strategy` promises. """


@pytest.mark.parametrize("module", parameters(MODULES))
@given(data=st.data())
def test_boundaries(module, data):
    """ A diagram has the boundary it is asked for, in its own category. """
    dom = data.draw(module.Ty.strategy(max_length=2), label="dom")
    cod = data.draw(module.Ty.strategy(max_length=2), label="cod")
    diagram = data.draw(
        module.Diagram.strategy(dom=dom, cod=cod), label="diagram")
    assert isinstance(diagram, module.Diagram)
    assert (diagram.dom, diagram.cod) == (dom, cod)


def leaves(diagram) -> int:
    """
    The number of generators in a diagram, counting those inside the
    bubbles of traces and feedback rather than the bubbles themselves.
    """
    return sum(
        leaves(box.arg) if hasattr(box, "arg") else 1
        for box in diagram.boxes)


@pytest.mark.parametrize("module", parameters(MODULES))
@given(data=st.data())
def test_leaves(module, data):
    """
    A diagram has at least ``min_leaves`` generators when asked to, before
    any normal form: snake removal may undo a cap followed by a cup.
    """
    params = dict(normal_form=False) if module in PLANAR else {}
    diagram = data.draw(module.Diagram.strategy(
        min_leaves=2, max_leaves=2, **params))
    assert leaves(diagram) >= 2


@pytest.mark.parametrize("module", parameters(PLANAR))
@given(data=st.data())
def test_planar_normal_form(module, data):
    """
    A planar diagram is generated in normal form, unless it has none
    because it is not boundary-connected.
    """
    diagram = data.draw(module.Diagram.strategy())
    try:
        normal = diagram.normal_form()
    except NotImplementedError:
        assert not diagram.to_hypergraph().is_boundary_connected
        return
    assert normal == diagram


@pytest.mark.parametrize("module", parameters(SYMMETRIC + MARKOV))
@given(data=st.data())
def test_hypergraph_spiders(module, data):
    """ A hypergraph meets the constraint on spiders of its category. """
    graph_factory = Hypergraph[module.Diagram]
    graph = data.draw(graph_factory.strategy())
    assert SPIDERS[graph_factory.default_spiders()](graph)


@pytest.mark.parametrize("module", parameters(SYMMETRIC + MARKOV))
@given(data=st.data())
def test_hypergraph_readback(module, data):
    """ A hypergraph reads back from the diagram it downgrades to. """
    graph = data.draw(Hypergraph[module.Diagram].strategy(), label="graph")
    diagram = graph.to_diagram()
    note(diagram)
    assert diagram.to_hypergraph() == graph


@pytest.mark.parametrize("module", parameters(SYMMETRIC))
@given(data=st.data())
def test_cmap_readback(module, data):
    """
    A map is monogamous unless its category has cups and caps, and it reads
    back from the diagram it downgrades to, up to the order of its boxes,
    i.e. as a hypergraph.
    """
    cmap = data.draw(CMap[module.Diagram].strategy(), label="cmap")
    assert cmap.is_monogamous or module is compact
    diagram = cmap.to_diagram()
    note(diagram)
    assert diagram.to_hypergraph() == cmap.to_hypergraph()


@pytest.mark.parametrize("spiders", list(SPIDERS))
@given(data=st.data())
def test_spider_constraints(spiders, data):
    """
    Every constraint of :meth:`Hypergraph.strategy` holds whatever the
    boundary, drawn in the most general category: frobenius diagrams.
    """
    x = data.draw(frobenius.Ty.strategy(max_length=2), label="dom")
    y = data.draw(frobenius.Ty.strategy(max_length=2), label="cod")
    acyclic = data.draw(st.booleans(), label="acyclic")
    graph = data.draw(Hypergraph[frobenius.Diagram].strategy(
        dom=x, cod=y, spiders=spiders, acyclic=acyclic), label="graph")
    assert (graph.dom, graph.cod) == (x, y)
    assert SPIDERS[spiders](graph)
    if acyclic and spiders in ("left_monogamous", "monogamous"):
        assert graph.is_acyclic or graph.scalar_spiders
