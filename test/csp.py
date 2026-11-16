import pytest

pytest.importorskip("opt_einsum")

import numpy as np
from hypothesis import given, settings, strategies as st
from pytest import raises

from discopy import frobenius
from discopy.csp import Language, instance, scopes

X, Y = frobenius.Ty('x'), frobenius.Ty('y')
SIZES = {X: 2, Y: 3}


@st.composite
def problems(draw, max_variables=4, max_constraints=4, max_arity=3):
    """ A random instance on ``x`` and ``y``, with random outputs, and a
    language with a random relation for each of its constraints. """
    sorts = draw(st.lists(
        st.sampled_from([X, Y]), min_size=1, max_size=max_variables))
    variables = st.integers(0, len(sorts) - 1)
    constraints = [(f"R{i}", tuple(scope)) for i, scope in enumerate(
        draw(st.lists(st.lists(variables, max_size=max_arity),
                      max_size=max_constraints)))]
    outputs = tuple(draw(st.lists(variables, max_size=2)))
    relations = {}
    for name, scope in constraints:
        size = int(np.prod([SIZES[sorts[i]] for i in scope]))
        table = draw(st.lists(st.booleans(), min_size=size, max_size=size))
        shape = [SIZES[sorts[i]] for i in scope]
        relations[name] = np.array(table, dtype=bool).reshape(shape)
    predicates = {
        name: (lambda array: lambda *values: bool(array[values]))(array)
        for name, array in relations.items()}
    return instance(constraints, sorts, outputs), Language(SIZES, predicates)


def test_instance_scopes_roundtrip():
    constraints = [("R", (0, 1)), ("S", (1, 1)), ("T", ()), ("R", (2, 0))]
    graph = instance(constraints, [X, Y, X], outputs=(2, 2))
    assert scopes(graph) == constraints
    assert instance(
        scopes(graph), graph.spider_types, graph.cod_wires) == graph
    renamed = instance(
        [("R", (1, 2)), ("S", (2, 2)), ("T", ()), ("R", (0, 1))],
        [X, X, Y], outputs=(0, 0))
    assert renamed == graph


def test_scopes_of_a_hypergraph_with_inputs():
    with raises(ValueError):
        scopes(frobenius.Box('f', X, X).to_hypergraph())


@settings(max_examples=60, deadline=None)
@given(problems())
def test_instance_scopes_roundtrip_random(problem):
    graph, _ = problem
    assert instance(
        scopes(graph), graph.spider_types, graph.cod_wires) == graph


@settings(max_examples=60, deadline=None)
@given(problems())
def test_solutions_are_counted_by_the_functor(problem):
    """ The number of solutions found by backtracking is the image of the
    closed instance over ``int``, and each one holds. """
    graph, language = problem
    closed = instance(scopes(graph), graph.spider_types)
    solutions = list(language.solutions(closed))
    assert all(language.holds(closed, solution) for solution in solutions)
    assert len(solutions) == len(set(solutions))
    assert language.functor(int)(closed).array.item() == len(solutions)


@settings(max_examples=60, deadline=None)
@given(problems())
def test_functor_projects_the_solutions(problem):
    """ The image of an open instance over ``bool`` is the projection of its
    solutions on its outputs. """
    graph, language = problem
    shape = [SIZES[graph.spider_types[i]] for i in graph.cod_wires]
    expected = np.zeros(shape, dtype=bool)
    for solution in language.solutions(graph):
        expected[tuple(solution[i] for i in graph.cod_wires)] = True
    result = language.functor(bool)(graph).array
    assert np.array_equal(np.asarray(result).reshape(shape), expected)


def test_holds():
    language = Language({X: 2}, {"eq": lambda a, b: a == b})
    graph = instance([("eq", (0, 1))], [X, X])
    assert language.holds(graph, (1, 1)) and not language.holds(graph, (0, 1))


def test_a_false_nullary_constraint_has_no_solution():
    language = Language({X: 2}, {"no": lambda: False})
    graph = instance([("no", ())], [X])
    assert list(language.solutions(graph)) == []
    assert language.functor(int)(graph).array.item() == 0
