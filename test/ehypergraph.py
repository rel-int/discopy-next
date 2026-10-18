import random
from functools import reduce
from itertools import islice
from operator import eq, lt

import numpy as np
import pytest
from hypothesis import assume, example, given, settings, strategies as st
from pytest import raises

from discopy import symmetric
from discopy.ehypergraph import EHypergraph, Markov, partition
from discopy.hypergraph import Hypergraph
from discopy.markov import Ty, Box, Copy, Diagram, Functor, Merge, Swap
from discopy.tensor import Dim, Tensor
from discopy.utils import AxiomError

x = Ty('x')
f, g, h = Box('f', x, x), Box('g', x, x), Box('h', x, x)
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
    unit = Markov.id(Ty()).add(Merge(x, 0), ())[0]
    assert unit.add(Merge(x, 0), ())[1] == (1, )


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
    assert Markov.from_diagram(f.dagger() >> Diagram.discard(x)).extract(
        ).boxes == (f.dagger(), )


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


def narrow(diagram: Diagram, bound: int = 12) -> bool:
    """ Whether each layer of a diagram has at most ``bound`` wires around. """
    return all(len(layer.dom @ layer.cod) <= bound for layer in diagram.inside)


def model(seed: int, alike: dict) -> Functor:
    """ A random functor into tensors of dimension two, a box as its class in
    ``alike``: a deterministic box a function, partial when not affine, any
    other a random matrix, stochastic when affine. """
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
            table.setdefault(alike.get(box, box), array(box)),
            Dim(*len(box.dom) * (2, )), Dim(*len(box.cod) * (2, )))

    return Functor(ob_map=lambda _: Dim(2), ar_map=ar_map, cod=Tensor[float])


@settings(max_examples=50, deadline=None)
@given(st.data())
def test_sections_evaluate_as_the_diagram(data):
    diagram = data.draw(Markov.strategy(
        effects=True, max_merges=0)).graph.to_diagram()
    if data.draw(st.booleans()):
        diagram = Diagram.copy(diagram.dom) >> diagram @ diagram
    assume(narrow(diagram, 16))
    graph = Markov(Hypergraph[Diagram].from_diagram(diagram))
    boxes, rewritten, related = graph.graph.boxes, graph, []
    pairs = st.tuples(st.integers(0, 9), st.integers(0, 9))
    for i, j in data.draw(st.lists(pairs, max_size=3)) if boxes else ():
        i, j = i % len(boxes), j % len(boxes)
        if boxes[i].is_cartesian and boxes[j].is_cartesian and (
                boxes[i].dom, boxes[i].cod) == (boxes[j].dom, boxes[j].cod):
            rewritten = rewritten.rewrite(
                (i, ), *graph.graph.box_wires[i], boxes[j].to_hypergraph())
            related.append((boxes.index(boxes[i]), boxes.index(boxes[j])))
    alike = dict(zip(boxes, map(boxes.__getitem__, partition(
        len(boxes), related))))
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
                 (f, g @ Diagram.discard(x)), (f, sample), (sample, f)]:
        with raises(AxiomError, match="is not a rule"):
            graph.saturate(rule)


def test_matches_bind_classes():
    divide, m = Box('/', x @ x, x), Box('m', x @ x, x)
    graph = Markov.from_diagram(divide)
    lhs = (Copy(x) >> divide).to_hypergraph()
    assert graph.matches(lhs) == []
    assert graph.merge(0, 1).matches(lhs) == [((0, ), (0, ), (2, ))]
    assert Markov.from_diagram(f @ g).matches((g >> f).to_hypergraph()) == []
    twins = Copy(x) @ Copy(x) >> x @ Swap(x, x) @ x\
        >> x @ x @ Swap(x, x) >> m @ m
    assert Markov.from_diagram(twins).matches(twins.to_hypergraph()) == [
        ((0, 1), (1, 0), (3, 2)), ((0, 1), (0, 1), (2, 3))]


def test_saturate_stops_and_rebuilds():
    graph = Markov.from_diagram(f >> f)
    rounds = list(graph.saturate((f, g), (g, f)))
    assert len(rounds) == 1 and len(rounds[0].graph.boxes) == 4
    assert rounds[0].extract().boxes == (f, f)
    *_, last = Markov.from_diagram(
        Copy(x) >> (g >> f) @ (h >> f)).saturate((g, h))
    first, second = last.graph.cod_wires
    assert last.classes[first] == last.classes[second]


def to_diagram(term, names) -> Diagram:
    """ The diagram of a term of :mod:`fixtures.egglog_reference` on inputs
    ``names``, a box for each subterm, e.g. ``("sample", p, id)`` and
    ``("observe", c, t)``, ``t`` conditioned on ``c``. """
    boxes, box_wires, built = [], [], {}

    def build(term):
        if term[0] == "var":
            return names.index(term[1])
        if term not in built:
            inputs, box = tuple(map(build, children(term))), node(term)
            outputs = (len(names) + len(boxes), )[:len(box.cod)]
            boxes.append(box)
            box_wires.append((inputs, outputs))
            built[term] = build(term[2]) if term[0] == "observe"\
                else outputs[0]
        return built[term]

    root = build(term)
    return Hypergraph[Diagram](
        x ** len(names), x, tuple(boxes),
        (tuple(range(len(names))), tuple(box_wires), (root, ))
    ).to_diagram()


def node(term) -> Box:
    """ The box of the head of a term that is not a variable. """
    if term[0] in ("sample", "observe"):
        return sample if term[0] == "sample" else condition
    return Box(str(term[1]), Ty(), x) if term[0] == "const"\
        else Box(term[0], x ** len(term[1:]), x)


def children(term) -> tuple:
    """ The subterms a box reads, not the id of a sample nor what follows. """
    if term[0] in ("var", "const"):
        return ()
    return term[1:2] if term[0] in ("sample", "observe") else term[1:]


def variables(term) -> list[str]:
    """ The variables of a term, in the order they first occur. """
    if term[0] in ("var", "const"):
        return [term[1]] if term[0] == "var" else []
    return list(dict.fromkeys(sum(map(variables, term[1:]), [])))


def to_rules(rules) -> list[tuple[Diagram, Diagram]]:
    """ The diagrams of rules of terms, over the variables of the left. """
    return [(to_diagram(lhs, variables(lhs)), to_diagram(rhs, variables(lhs)))
            for lhs, rhs in rules]


def box_cost(box) -> int:
    """ The cost of a box in :mod:`fixtures.egglog_reference`. """
    from fixtures.egglog_reference import COSTS

    return COSTS.get(box.name, 1)


def lookup(graph: EHypergraph, term, names) -> int:
    """ The class of a subterm of the cartesian term a graph comes from. """
    if term[0] == "var":
        return graph.classes[graph.graph.dom_wires[names.index(term[1])]]
    inputs = tuple(lookup(graph, child, names) for child in children(term))
    return next(graph.classes[tgt[0]] for j, (box, (_, tgt)) in enumerate(zip(
        graph.graph.boxes, graph.graph.box_wires))
        if box == node(term) and graph.inputs(j) == inputs)


def trace(term, rules, rounds: int) -> list[tuple]:
    """ For each round of saturating a term, as ``egglog_trace`` gives it:
    whether it changed anything, the classes of the subterms and the cost of
    what is extracted, under the costs of the reference. """
    from fixtures.egglog_reference import subterms

    graph = Markov.from_diagram(to_diagram(term, variables(term)))
    states = [graph, *islice(graph.saturate(*to_rules(rules)), rounds)]
    result = []
    for k in range(min(len(states) + 1, rounds + 1)):
        state, members = states[min(k, len(states) - 1)], {}
        for sub in subterms(term):
            members.setdefault(lookup(state, sub, variables(term)), []).append(
                sub)
        result.append((k < len(states), tuple(sorted((tuple(sorted(
            group, key=repr)) for group in members.values()), key=repr)),
            Markov(state.extract(box_cost, 0)).cost(box_cost, 0)))
    return result


A, C0, C1, C2 = ("var", "a"), ("const", 0), ("const", 1), ("const", 2)


@pytest.mark.parametrize("term, rounds", [
    (("/", ("*", A, C2), C2), 5),
    (("+", ("<<", ("*", A, C1), C2), ("<<", A, C2)), 3)])
def test_egglog_example(term, rounds):
    pytest.importorskip("egglog")
    from fixtures.egglog_reference import EGG, egglog_trace

    expected = [(state.updated, state.classes, state.cost)
                for state in egglog_trace(term, EGG, 10)]
    assert trace(term, EGG, 10) == expected and len(expected) == rounds


@pytest.mark.parametrize("seed", range(30))
def test_egglog_differential(seed):
    """ Our trace and egglog's on a random term of depth three or four and
    one of the rule sets of the reference, chosen by a seed. """
    pytest.importorskip("egglog")
    from fixtures.egglog_reference import RULESETS, egglog_trace, random_term

    rng = random.Random(seed)
    term, rules = random_term(rng, rng.choice((3, 4))), rng.choice(
        list(RULESETS.values()))
    assert trace(term, rules, 4) == [
        (state.updated, state.classes, state.cost)
        for state in egglog_trace(term, rules, 4)]


X, Y, P = ("var", "x"), ("var", "y"), ("var", "p")
CANCEL, ABSORB = [(("-", X, X), C0)], [(("*", C0, X), C0)]
MOD3 = CANCEL + ABSORB + [
    (("*", X, C0), C0), (("+", X, C0), X), (("*", X, C1), X),
    (("+", X, Y), ("+", Y, X)), (("*", X, Y), ("*", Y, X)),
    (("-", ("+", X, Y), Y), X)]
S0, S1 = (("sample", P, ("var", f"id{i}")) for i in range(2))
TWO, ONE, OBSERVED = ("-", S0, S1), ("-", S0, S0), (
    "observe", S0, ("*", C0, S0))
IDS = st.sampled_from([("var", f"id{i}") for i in range(3)])
TERMS = st.recursive(st.sampled_from([P, C0, C1, C2]), lambda terms: st.tuples(
    st.sampled_from("+-*"), terms, terms) | st.tuples(
    st.just("sample"), terms, IDS), max_leaves=6)
PROGRAMS = st.builds(lambda value, conditions: reduce(
    lambda program, c: ("observe", c, program), conditions, value),
    TERMS, st.lists(TERMS, max_size=2))


def arithmetic(seed: int, n: int = 3) -> Functor:
    """ A functor reading ``x`` as the integers modulo ``n``, where ``MOD3``
    holds, a sample as a random stochastic matrix and a condition as a random
    weight, zero or one, zero somewhere. """
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(n) for b in range(n)]
    arrays = {str(c): np.eye(n)[[c]] for c in range(n)} | {
        op: np.eye(n)[[function(a, b) % n for a, b in pairs]] for op, function
        in zip("+-*", (int.__add__, int.__sub__, int.__mul__))} | {
        'sample': rng.dirichlet(np.ones(n), size=n),
        'condition': rng.permutation(np.arange(n) > 0)[:, None] * 1.}
    return Functor(ob_map=lambda _: Dim(n), ar_map=lambda box: Tensor[float](
        arrays[box.name], Dim(n) ** len(box.dom), Dim(n) ** len(box.cod)),
        cod=Tensor[float])


def naive(term):
    """ A term without the ids of its samples nor its conditions. """
    if term[0] == "observe":
        return naive(term[2])
    return term if term[0] in ("var", "const")\
        else (term[0], *map(naive, children(term)))


def saturated(program, rules) -> EHypergraph:
    """ The last round of saturating a program of ``p`` under some rules. """
    graph = Markov.from_diagram(to_diagram(program, ["p"]))
    *_, last = graph, *graph.saturate(*to_rules(rules))
    return last


@settings(max_examples=40, deadline=None)
@given(program=PROGRAMS, seed=st.integers(0, 2 ** 32 - 1))
@example(program=TWO, seed=0)
@example(program=ONE, seed=0)
@example(program=OBSERVED, seed=0)
def test_sampling_and_conditioning(program, seed):
    """ Every section of two rounds of a random program, its samples shared
    or not, under rules modulo three evaluates there as the program does,
    the e-hypergraph and the diagrams small enough to enumerate. """
    diagram, functor = to_diagram(program, ["p"]), arithmetic(seed)
    assume(narrow(diagram))
    graph, expected = Markov.from_diagram(diagram), functor(diagram).array
    for state in [graph, *islice(graph.saturate(*to_rules(MOD3)), 2)]:
        sections = state.sections() if len(state.graph.boxes) < 20 else ()
        for section in filter(narrow, (s.to_diagram() for s in sections)):
            assert np.allclose(functor(section).array, expected)


@pytest.mark.parametrize("program, rules, egg, compare", [
    (TWO, CANCEL, False, eq), (ONE, CANCEL, False, eq),
    (OBSERVED, ABSORB, False, eq),
    (("observe", S0, ("-", ("/", ("*", S0, C2), C2), S0)), CANCEL, True, eq),
    (("observe", S0, ("observe", S0, S0)), CANCEL, False, lt),
    (("observe", ("*", C0, S0), ("observe", C0, S0)), ABSORB, False, lt)])
def test_egglog_sampling(program, rules, egg, compare):
    """
    egglog extracts the faithful encoding of a program, an id for each sample
    and its conditions in the term, at our cost but where a condition repeats
    on related inputs: a hard constraint is idempotent, paid once here.
    """
    pytest.importorskip("egglog")
    from fixtures.egglog_reference import EGG, egglog_extract

    rules = rules + (EGG if egg else [])
    last = saturated(program, rules)
    assert compare(Markov(last.extract(box_cost, 0)).cost(box_cost, 0),
                   egglog_extract(program, rules)[1])


def test_egglog_naive_sampling():
    """
    The naive encoding is unsound: a sample a function, egglog cancels two
    samples, a program its value, it drops a condition registered beside it,
    both told apart from the program, which what we extract evaluates as.
    """
    pytest.importorskip("egglog")
    from fixtures.egglog_reference import egglog_extract

    functor = arithmetic(0)
    for program, rules, extra in [(TWO, CANCEL, ()), (
            OBSERVED, ABSORB, [("condition", naive(S0))])]:
        expected = functor(to_diagram(program, ["p"])).array
        extracted, _ = egglog_extract(naive(program), rules, extra=extra)
        assert extracted == C0 and not np.allclose(
            functor(to_diagram(extracted, ["p"])).array, expected)
        assert np.allclose(functor(saturated(
            program, rules).extract().to_diagram()).array, expected)
