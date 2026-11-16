import itertools

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from pytest import raises

from discopy import frobenius
from discopy.csp import Language, instance, scopes, sudoku, queens, cnf, maze

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


@settings(max_examples=60, deadline=None)
@given(problems(), st.data())
def test_a_box_bent_to_have_inputs_is_the_same_constraint(problem, data):
    """ Bending the first legs of each box into inputs, as a diagram drawn
    in the plane does, changes neither the solutions nor their count. """
    graph, language = problem
    sorts = graph.spider_types
    boxes, box_wires = [], []
    for name, scope in scopes(graph):
        k = data.draw(st.integers(0, len(scope)))
        boxes.append(frobenius.Box(
            name, frobenius.Ty().tensor(*(sorts[i] for i in scope[:k])),
            frobenius.Ty().tensor(*(sorts[i] for i in scope[k:]))))
        box_wires.append((scope[:k], scope[k:]))
    bent = frobenius.Hypergraph(
        frobenius.Ty(), graph.cod, tuple(boxes),
        ((), tuple(box_wires), graph.cod_wires), spider_types=sorts)
    assert scopes(bent) == scopes(graph)
    assert sorted(language.solutions(bent)) == sorted(
        language.solutions(graph))
    assert np.array_equal(
        np.asarray(language.functor(int)(bent).array),
        np.asarray(language.functor(int)(graph).array))


def test_holds_refuses_an_assignment_off_the_domains():
    language = Language({X: 2}, {"eq": lambda a, b: a == b})
    graph = instance([("eq", (0, 1))], [X, X])
    assert not language.holds(graph, (2, 2))
    assert not language.holds(graph, (1, 1, 1))
    assert not language.holds(graph, (1, ))


def test_a_daggered_box_is_its_relation_reversed():
    """ The dagger of a box reads its outputs then its inputs, as the
    functor does, which evaluates the box and transposes it. """
    language = Language({X: 2}, {"lt": lambda a, b: a < b})
    lt = frobenius.Box("lt", X, X)
    graph = (frobenius.Spider(0, 1, X) >> lt.dagger()).to_hypergraph()
    assert scopes(graph) == [("lt", (1, 0))]
    assert list(language.solutions(graph)) == [(1, 0)]
    assert np.array_equal(
        np.asarray(language.functor(bool)(graph).array), [True, False])


def test_sudoku_9x9_sizes():
    """ The 9x9 sudoku has 81 variables, 27 scopes of 9 and 810 edges in
    the primal graph of its scopes, each cell with 20 peers. """
    from discopy.neural.wiring import primal
    graph, _ = sudoku(3)
    constraints = scopes(graph)
    assert len(graph.spider_types) == 81 and len(constraints) == 27
    assert {len(scope) for _, scope in constraints} == {9}
    peers = primal([scope for _, scope in constraints], 81)
    assert {len(row) for row in peers} == {20}
    assert sum(map(len, peers)) // 2 == 810


def test_sudoku_4x4_counts_288():
    graph, language = sudoku(2)
    assert language.functor(int)(graph).array.item() == 288


@pytest.mark.parametrize("n, count", [(4, 2), (5, 10), (6, 4)])
def test_queens_counts(n, count):
    """ 2, 10 and 4 solutions: by the functor up to 5, by backtracking at 6,
    whose contraction takes seconds. """
    graph, language = queens(n)
    assert len(list(language.solutions(graph))) == count
    if n <= 5:
        assert language.functor(int)(graph).array.item() == count


def test_queens_has_no_padding():
    graph, _ = queens(5)
    arities = sorted(len(scope) for _, scope in scopes(graph))
    assert arities == 4 * [2] + 4 * [3] + 4 * [4] + 12 * [5]


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 4).flatmap(lambda n: st.lists(st.lists(
    st.integers(1, n).flatmap(lambda i: st.sampled_from([i, -i])),
    min_size=0, max_size=3), max_size=4)))
def test_cnf_counts_the_truth_table(clauses):
    """ A formula has as many solutions as its truth table has satisfying
    rows. """
    graph, language = cnf(clauses)
    n = max((abs(literal) for clause in clauses for literal in clause),
            default=0)
    rows = sum(all(any(row[abs(i) - 1] == (i > 0) for i in clause)
                   for clause in clauses)
               for row in itertools.product((False, True), repeat=n))
    assert language.functor(int)(graph).array.item() == rows
    assert len(list(language.solutions(graph))) == rows


def test_cnf_needs_negation_to_be_distinct():
    """ With ``Not`` read as equality, the counting law fails: ``x`` and
    ``not x`` have no solution, but one when the fresh bit equals ``x``. """
    graph, language = cnf([[1], [-1]])
    assert language.functor(int)(graph).array.item() == 0
    equality = Language(language.sizes, dict(
        language.predicates, Not=lambda bit, other: bit == other))
    assert equality.functor(int)(graph).array.item() == 1


MAZE = ["S..#.",
        "##.#.",
        ".....",
        ".###.",
        "..G#."]


def test_maze_has_one_solution_the_path():
    graph, language = maze(MAZE)
    assert len(graph.boxes) == 17 and len(graph.spider_types) == 16
    assert all(len(ports) == 2 for ports in map(
        lambda wires: wires[0] | wires[1], graph.spider_wires))
    (solution, ) = language.solutions(graph)
    on_path = [any(solution[i] for i in scope) for _, scope in scopes(graph)]
    assert sum(solution) == 10 and sum(on_path) == 11
    assert language.functor(int)(graph).array.item() == 1


def test_maze_with_a_cycle_takes_loops_beside_the_path():
    """ With a cycle off the path, the constraints also take the path with
    the loop beside it, as the docstring of :func:`maze` says. """
    graph, language = maze(["S.#..",
                            "#.#..",
                            "#.G##"])
    assert language.functor(int)(graph).array.item() == 2


@settings(max_examples=60, deadline=None)
@given(problems())
def test_propagate_removes_no_value_a_solution_takes(problem):
    graph, language = problem
    candidates = language.propagate(graph)
    for solution in language.solutions(graph):
        assert all(value in candidates[i] for i, value in enumerate(solution))


@st.composite
def trees(draw):
    """ An instance whose incidence graph, a node per variable and per
    constraint, is a tree: each new constraint has at most one variable
    already in the tree, and each variable is in it once. """
    n_constraints = draw(st.integers(1, 5))
    sorts, constraints = [draw(st.sampled_from([X, Y]))], []
    for i in range(n_constraints):
        anchor = draw(st.integers(0, len(sorts) - 1))
        fresh = draw(st.integers(0 if i else 1, 2))
        scope = (anchor, *range(len(sorts), len(sorts) + fresh))
        sorts += [draw(st.sampled_from([X, Y])) for _ in range(fresh)]
        constraints.append((f"R{i}", scope))
    relations = {}
    for name, scope in constraints:
        shape = [SIZES[sorts[i]] for i in scope]
        table = draw(st.lists(st.booleans(), min_size=int(np.prod(shape)),
                              max_size=int(np.prod(shape))))
        relations[name] = np.array(table, dtype=bool).reshape(shape)
    predicates = {
        name: (lambda array: lambda *values: bool(array[values]))(array)
        for name, array in relations.items()}
    return instance(constraints, sorts), Language(SIZES, predicates)


@settings(max_examples=300, deadline=None)
@given(trees())
def test_propagate_on_a_tree_is_the_projection_of_the_solutions(problem):
    """ On a tree, arc consistency at its fixed point leaves exactly the
    values some solution takes, or nothing when there is no solution. """
    graph, language = problem
    solutions = list(language.solutions(graph))
    projections = [{solution[i] for solution in solutions}
                   for i in range(len(graph.spider_types))]
    candidates = language.propagate(graph)
    if solutions:
        assert candidates == projections
    else:
        assert any(not values for values in candidates)


def test_propagate_needs_more_than_one_round():
    """ On a chain of ``lt`` read from its end, one round leaves values no
    solution takes: the tree law is for the fixed point. """
    language = Language({X: 3}, {"lt": lambda a, b: a < b})
    chain = instance([("lt", (1, 2)), ("lt", (0, 1))], 3 * [X])
    assert scopes(chain) == [("lt", (0, 1)), ("lt", (2, 0))]
    fixed = language.propagate(chain)
    assert fixed == [{1}, {2}, {0}]
    once = language.propagate(chain, rounds=1)
    assert once == [{1}, {1, 2}, {0}] and once != fixed


def test_propagate_alone_solves_the_maze():
    graph, language = maze(MAZE)
    (solution, ) = language.solutions(graph)
    assert language.propagate(graph) == [{value} for value in solution]
