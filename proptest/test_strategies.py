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
from discopy.axioms import inhabits
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


SHAPES = {
    symmetric: lambda graph: graph.is_monogamous,
    compact: lambda graph: graph.is_bijective,
    markov: lambda graph: graph.is_left_monogamous,
    closed: lambda graph: graph.is_left_monogamous,
    frobenius: lambda graph: True,
}
""" The shape of the hypergraphs each category generates. """


@pytest.mark.parametrize("module", parameters(MODULES))
@given(data=st.data())
def test_boundaries(module, data):
    """ A diagram has the boundary it is asked for, in its own category. """
    dom = data.draw(module.Ty.strategy(), label="dom")
    cod = data.draw(module.Ty.strategy(), label="cod")
    diagram = data.draw(
        module.Diagram.strategy(dom=dom, cod=cod), label="diagram")
    assert isinstance(diagram, module.Diagram)
    assert (diagram.dom, diagram.cod) == (dom, cod)


RULES = [
    pytest.param(module, name, id=f"{factory_name(module.Diagram)}.{name}")
    for module in MODULES for name in module.Diagram.rules]
""" Every rule of every level, with its level. """


@pytest.mark.parametrize("module, name", RULES)
@given(data=st.data())
def test_rule(module, name, data):
    """
    A rule applies to the premises it draws for a goal and concludes a
    diagram of the type its signature states, in the scope of the draw.
    """
    rule = module.Diagram.rules[name]
    dom = data.draw(module.Ty.strategy(), label="dom")
    cod = data.draw(module.Ty.strategy(), label="cod")
    scope, premises = data.draw(
        rule.sequents(rule.given(dom, cod), fuel=1), label="premises")
    assert inhabits(rule(**premises), rule.output(scope))


@pytest.mark.parametrize("module", parameters(SYMMETRIC + MARKOV))
@given(data=st.data())
def test_hypergraph_shape(module, data):
    """
    A hypergraph has the boundary it is asked for and the shape of its
    category, whatever the boundary, and is acyclic when asked to be in a
    category whose spiders have a producer.
    """
    dom = data.draw(module.Ty.strategy(), label="dom")
    cod = data.draw(module.Ty.strategy(), label="cod")
    acyclic = data.draw(st.booleans(), label="acyclic")
    graph = data.draw(Hypergraph[module.Diagram].strategy(
        dom=dom, cod=cod, acyclic=acyclic), label="graph")
    assert (graph.dom, graph.cod) == (dom, cod)
    assert SHAPES[module](graph)
    if acyclic and module in (symmetric, markov, closed):
        assert graph.is_acyclic


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
