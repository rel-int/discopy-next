import random
from itertools import islice

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from pytest import raises

from discopy import symmetric
from discopy.ehypergraph import EHypergraph, Markov
from discopy.hypergraph import Hypergraph
from discopy.markov import Ty, Box, Copy, Diagram, Functor
from discopy.tensor import Dim, Tensor
from discopy.utils import AxiomError

x = Ty('x')
f, g = Box('f', x, x), Box('g', x, x)
sample = Box('sample', x, x, is_deterministic=False)
condition = Box('condition', x, Ty(), is_affine=False)
score = Box('score', x, Ty(), is_deterministic=False, is_affine=False)


def test_samples_are_independent():
    assert len(set(Markov.from_diagram(Copy(x) >> f @ f).classes)) == 2
    assert len(set(Markov.from_diagram(Copy(x) >> f @ g).classes)) == 3
    assert len(set(Markov.from_diagram(
        Copy(x) >> sample @ sample).classes)) == 3
    graph = Markov.from_diagram(sample)
    assert len(graph.add(sample, (0, ))[0].graph.boxes) == 2
    assert Markov.from_diagram(f).add(f, (0, ))[1] == (1, )


def test_category_has_flags():
    with raises(AxiomError):
        EHypergraph[symmetric.Diagram].from_diagram(symmetric.Box('f', x, x))


def test_effects_are_kept():
    hard = Markov.from_diagram(Copy(x, 3) >> condition @ condition @ x)
    assert hard.roots() == (0, ) and hard.extract().boxes == (condition, )
    soft = Markov.from_diagram(Copy(x, 3) >> score @ score @ x)
    assert soft.roots() == (0, 1) and soft.extract().boxes == (score, score)
    unread = Markov.from_diagram(f >> condition)
    assert unread.extract().boxes == (f, condition) and unread.cost() == 3
    assert Markov.from_diagram(f >> Diagram.discard(x)).extract().boxes == ()


def test_rewrite_refusals():
    graph = Markov.from_diagram(sample >> f)
    merge = Hypergraph[Diagram](x @ x, x, (), ((0, 0), (), (0, )))
    cyclic = Markov.from_diagram(f).merge(0, 1)
    with raises(AxiomError):
        cyclic.rewrite((0, ), (), (1, ), Box('c', Ty(), x).to_hypergraph())
    for region, dom, cod, rhs in [
            ((0, ), (0, ), (1, ), f.to_hypergraph()),
            ((1, ), (1, ), (2, ), sample.to_hypergraph()),
            ((1, ), (0, ), (2, ), g.to_hypergraph()),
            ((1, ), (1, ), (2, ), Box('h', x @ x, x).to_hypergraph()),
            ((1, ), (1, 1), (2, ), merge)]:
        with raises(AxiomError):
            graph.rewrite(region, dom, cod, rhs)


def test_rewrite_glues():
    graph = Markov.from_diagram(f >> g)
    rewritten = graph.rewrite(
        (0, 1), (0, ), (2, ), Diagram.id(x).to_hypergraph())
    assert len(rewritten.graph.boxes) == 2 and rewritten.classes == (0, 1, 0)
    assert rewritten.extract() == Diagram.id(x).to_hypergraph()
    assert len(rewritten.sections()) == 1


def find(alike: dict, box: Box) -> Box:
    """ The representative of a box, as the union-find ``alike`` says. """
    while box in alike:
        box = alike[box]
    return box


def model(seed: int, alike: dict) -> Functor:
    """
    A random functor into tensors with a dimension two for each wire, which
    interprets two boxes alike when ``alike`` relates them: a deterministic
    box is a function, partial when it is not affine, any other a random
    matrix, stochastic when affine.
    """
    rng, table = np.random.default_rng(seed), {}

    def array(box):
        rows, columns = 2 ** len(box.dom), 2 ** len(box.cod)
        if box.is_deterministic:
            return np.eye(columns)[rng.integers(columns, size=rows)] * (
                1 if box.is_affine else rng.integers(2, size=(rows, 1)))
        matrix = rng.random((rows, columns))
        return matrix / matrix.sum(axis=1, keepdims=True)\
            if box.is_affine else matrix

    def ar_map(box):
        return Tensor[float](
            table.setdefault(find(alike, box), array(box)),
            Dim(*len(box.dom) * (2, )), Dim(*len(box.cod) * (2, )))

    return Functor(ob_map=lambda _: Dim(2), ar_map=ar_map, cod=Tensor[float])


@settings(max_examples=50, deadline=None)
@given(st.data())
def test_sections_evaluate_as_the_diagram(data):
    diagram = data.draw(Markov.strategy(
        effects=True, max_merges=0)).graph.to_diagram()
    if data.draw(st.booleans()):
        diagram = Diagram.copy(diagram.dom) >> diagram @ diagram
    graph = Markov(Hypergraph[Diagram].from_diagram(diagram))
    boxes, rewritten, alike = graph.graph.boxes, graph, {}
    pairs = st.tuples(st.integers(0, 9), st.integers(0, 9))
    for i, j in data.draw(st.lists(pairs, max_size=3)) if boxes else ():
        i, j = i % len(boxes), j % len(boxes)
        if boxes[i].is_cartesian and boxes[j].is_cartesian and (
                boxes[i].dom, boxes[i].cod) == (boxes[j].dom, boxes[j].cod):
            rewritten = rewritten.rewrite(
                (i, ), *graph.graph.box_wires[i], boxes[j].to_hypergraph())
            if find(alike, boxes[i]) != find(alike, boxes[j]):
                alike[find(alike, boxes[i])] = find(alike, boxes[j])
    functor = model(data.draw(st.integers(0, 2 ** 32 - 1)), alike)
    expected = functor(diagram).array
    for section in rewritten.rebuild().sections():
        assert np.allclose(functor(section.to_diagram()).array, expected)


def test_matches_flags_and_patterns():
    noisy = Markov.from_diagram(Copy(x) >> sample @ Box('sample', x, x))
    assert noisy.matches(Box('sample', x, x).to_hypergraph()) == [
        ((1, ), (0, ), (2, ))]
    graph = Markov.from_diagram(x @ Box('c', Ty(), x) >> Box('h', x @ x, x))
    unread = Hypergraph[Diagram](x, x, (Box('c', Ty(), x), ), ((0, ), (
        ((), (1, )), ), (1, )))
    assert graph.matches(unread) == [((0, ), (None, ), (1, ))]
    assert graph.rewrite((0, ), (None, ), (1, ), unread) == graph
    for lhs in [Diagram.id(x).to_hypergraph(), sample.to_hypergraph()]:
        with raises(AxiomError):
            graph.matches(lhs)
    with raises(AxiomError):
        graph.rewrite((0, ), (None, ), (1, ), f.to_hypergraph())
    for rule in [(Diagram.discard(x) >> Box('c', Ty(), x), f),
                 (f, g @ Diagram.discard(x)), (f, sample)]:
        with raises(AxiomError):
            next(graph.saturate(rule))


def test_saturate_stops():
    graph = Markov.from_diagram(f >> f)
    rounds = list(graph.saturate((f, g), (g, f)))
    assert len(rounds) == 1 and len(rounds[0].graph.boxes) == 4
    assert rounds[0].extract().boxes == (f, f)


def to_diagram(term, variables) -> Diagram:
    """
    The cartesian diagram of a term of :mod:`fixtures.egglog_reference`, one
    box for each operation or constant and one domain wire for each variable,
    copied as many times as it occurs.
    """
    boxes, box_wires = [], []

    def build(term):
        if term[0] == "var":
            return variables.index(term[1])
        inputs = tuple(map(build, children(term)))
        boxes.append(node(term))
        box_wires.append((inputs, (len(variables) + len(boxes) - 1, )))
        return box_wires[-1][1][0]

    root = build(term)
    return Hypergraph[Diagram](
        x ** len(variables), x, tuple(boxes),
        (tuple(range(len(variables))), tuple(box_wires), (root, ))
    ).to_diagram()


def node(term) -> Box:
    """ The box of the head of a term that is not a variable. """
    return Box(str(term[1]), Ty(), x) if term[0] == "const"\
        else Box(term[0], x ** len(term[1:]), x)


def children(term) -> tuple:
    return () if term[0] in ("var", "const") else term[1:]


def lookup(graph: EHypergraph, term, variables) -> int:
    """ The class of a subterm of the term an e-hypergraph comes from. """
    if term[0] == "var":
        return graph.classes[graph.graph.dom_wires[variables.index(term[1])]]
    inputs = tuple(lookup(graph, child, variables) for child in children(term))
    return next(graph.classes[tgt[0]] for j, (box, (_, tgt)) in enumerate(zip(
        graph.graph.boxes, graph.graph.box_wires))
        if box == node(term) and graph.inputs(j) == inputs)


def trace(term, rules, rounds: int) -> list[tuple]:
    """
    For each round of :meth:`EHypergraph.saturate`, as
    :func:`fixtures.egglog_reference.egglog_trace` gives them: whether it
    changed anything, the partition of the subterms of ``term`` and the cost
    of what is extracted, under egglog's costs.
    """
    from fixtures.egglog_reference import subterms

    def variables(term):
        return list(dict.fromkeys(
            sub[1] for sub in subterms(term) if sub[0] == "var"))

    graph = Markov.from_diagram(to_diagram(term, variables(term)))
    states = [graph, *islice(graph.saturate(*(
        (to_diagram(lhs, variables(lhs)), to_diagram(rhs, variables(lhs)))
        for lhs, rhs in rules)), rounds)]
    result = []
    for k in range(min(len(states) + 1, rounds + 1)):
        state, members = states[min(k, len(states) - 1)], {}
        for sub in subterms(term):
            members.setdefault(lookup(state, sub, variables(term)), []).append(
                sub)
        result.append((k < len(states), tuple(sorted((tuple(sorted(
            group, key=repr)) for group in members.values()), key=repr)),
            Markov(state.extract(input_cost=0)).cost(input_cost=0)))
    return result


def differential(seed: int, rounds: int = 4) -> tuple[list, list]:
    """
    Our trace and egglog's on a random term of depth three or four and one
    of the rule sets of the reference, chosen by a seed.
    """
    from fixtures.egglog_reference import RULESETS, egglog_trace, random_term

    rng = random.Random(seed)
    term, rules = random_term(rng, rng.choice((3, 4))), rng.choice(
        list(RULESETS.values()))
    return trace(term, rules, rounds), [
        (state.updated, state.classes, state.cost)
        for state in egglog_trace(term, rules, rounds)]


def test_egglog_example():
    pytest.importorskip("egglog")
    from fixtures.egglog_reference import EGG, C2, egglog_trace

    term = ("/", ("*", ("var", "a"), C2), C2)
    expected = [(state.updated, state.classes, state.cost)
                for state in egglog_trace(term, EGG, 10)]
    assert trace(term, EGG, 10) == expected and len(expected) == 5


@pytest.mark.parametrize("seed", range(30))
def test_egglog_differential(seed):
    pytest.importorskip("egglog")
    ours, egglog = differential(seed)
    assert ours == egglog
