"""
A constraint satisfaction problem as a diagram, and its language as a functor.

An instance is an arrow of the free hypergraph category on a signature: a
variable is a spider, a constraint a box with no input plugged on its scope.
A closed instance is a yes or no question, an open one a conjunctive query
whose outputs are its free variables, the other spiders being quantified
away :cite:p:`ChandraMerlin77,BonchiEtAl18,FeliceEtAl20`. Concretely, an
instance is a :class:`frobenius.Hypergraph` whose boxes have an empty
domain: the scope of a constraint is its entry in ``box_wires``, and
equality is up to renaming the variables.

A language sends a sort to a finite set, ``range(size)``, and a symbol to a
relation, given by a Python predicate on values. It is a hypergraph functor
into finite relations, i.e. :class:`tensor.Tensor` over ``bool``, which
sends an instance to its set of solutions projected on its outputs; the
same arrays over ``int`` count the solutions.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    instance
    scopes
    all_different
    exactly_one
    at_most_one
    zero_or_two
    disjunction
    negation
    sudoku
    queens
    cnf
    maze_diagram
    maze

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Language

Example
-------
>>> x = frobenius.Ty('x')
>>> graph = instance([("neq", (0, 1)), ("neq", (1, 2))], 3 * [x])
>>> language = Language({x: 2}, {"neq": lambda a, b: a != b})
>>> sorted(language.solutions(graph))
[(0, 1, 0), (1, 0, 1)]
>>> language.functor(int)(graph)
Tensor[int]([2], dom=Dim(1), cod=Dim(1))
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product

from discopy import cat, frobenius, monoidal, tensor
from discopy.utils import assert_isinstance

Constraint = tuple[str, tuple[int, ...]]
""" A constraint symbol and its scope, the variables it is plugged on. """


def instance(
        constraints: Sequence[Constraint], sorts: Sequence[frobenius.Ty],
        outputs: Sequence[int] = ()) -> frobenius.Hypergraph:
    """
    The instance with a variable of each sort, a box for each constraint
    plugged on its scope and the given variables as its outputs.

    A hypergraph is equal to any renaming of its spiders, and numbers them
    as they first occur on its ports: the scopes in order, then its
    outputs, then the variables on no port. So the variables keep their
    numbers when they first occur in that order, as in this example, and
    are renamed otherwise.

    Parameters:
        constraints : A symbol and a scope for each constraint.
        sorts : The sort of each variable, an atomic type.
        outputs : The free variables, in the order of the codomain.

    Example
    -------
    >>> x, y = map(frobenius.Ty, "xy")
    >>> graph = instance([("R", (0, 1)), ("S", (1, ))], [x, y], (1, ))
    >>> graph.boxes
    (frobenius.Box('R', frobenius.Ty(), frobenius.Ty(frobenius.Wire('x'), \
frobenius.Wire('y'))), frobenius.Box('S', frobenius.Ty(), \
frobenius.Ty(frobenius.Wire('y'))))
    >>> assert scopes(graph) == [("R", (0, 1)), ("S", (1, ))]
    >>> scopes(instance([("R", (1, 0))], [y, x]))
    [('R', (0, 1))]
    """
    sorts = tuple(sorts)
    for sort in sorts:
        assert_isinstance(sort, frobenius.Ty)
    boxes = tuple(frobenius.Box(name, frobenius.Ty(), frobenius.Ty().tensor(
        *(sorts[variable] for variable in scope)))
        for name, scope in constraints)
    cod = frobenius.Ty().tensor(*(sorts[variable] for variable in outputs))
    box_wires = tuple(((), tuple(scope)) for _, scope in constraints)
    return frobenius.Hypergraph(
        frobenius.Ty(), cod, boxes, ((), box_wires, tuple(outputs)),
        spider_types=dict(enumerate(sorts)))


def scopes(graph: frobenius.Hypergraph) -> list[Constraint]:
    """
    The symbol and the scope of each constraint of an instance, so that
    ``instance(scopes(graph), graph.spider_types, graph.cod_wires)`` is the
    instance itself when its boxes have no input.

    A box may also have inputs, bent into outputs as a hypergraph category
    allows: its scope is the variables on its inputs, then on its outputs,
    so that a diagram drawn in the plane, e.g. a maze with a wire per edge,
    reads as the instance it draws. Rebuilt by :func:`instance`, its boxes
    have no input: the scopes are the same, the hypergraph is not. The
    dagger of a box is its relation reversed: its scope is its outputs,
    then its inputs, those of the box it is the dagger of.

    Parameters:
        graph : An instance, a hypergraph with no input.

    Example
    -------
    >>> x = frobenius.Ty('x')
    >>> neq = frobenius.Box("neq", x, x)
    >>> chain = (frobenius.Spider(0, 1, x) >> neq >> neq).to_hypergraph()
    >>> scopes(chain)
    [('neq', (0, 1)), ('neq', (1, 2))]
    >>> language = Language({x: 3}, {"neq": lambda a, b: a != b})
    >>> language.functor(int)(chain)
    Tensor[int]([4, 4, 4], dom=Dim(1), cod=Dim(3))
    """
    if graph.dom:
        raise ValueError(
            f"Expected an instance, with no inputs, got {graph!r}.")
    return [(box.name, cod_wires + dom_wires if box.is_dagger
             else dom_wires + cod_wires)
            for box, (dom_wires, cod_wires) in zip(
                graph.boxes, graph.box_wires)]


@dataclass
class Language:
    """
    A language: the size of each sort, whose values are ``range(size)``,
    and the predicate of each symbol, which takes one value per variable of
    a scope and says whether the constraint holds.

    Parameters:
        sizes : The number of values of each sort.
        predicates : The relation of each symbol, as a predicate.
        arrays : The relation of each box enumerated so far, see
            :meth:`array`.

    Example
    -------
    >>> x = frobenius.Ty('x')
    >>> language = Language({x: 3}, {"lt": lambda a, b: a < b})
    >>> graph = instance([("lt", (0, 1))], [x, x], outputs=(1, ))
    >>> language.holds(graph, (0, 2)), language.holds(graph, (2, 0))
    (True, False)
    >>> language.functor(bool)(graph)
    Tensor[bool]([False, True, True], dom=Dim(1), cod=Dim(3))
    """
    sizes: Mapping[monoidal.Ty, int]
    predicates: Mapping[str, Callable[..., bool]]
    arrays: dict = field(default_factory=dict, repr=False, compare=False)

    def domain(self, sort: monoidal.Ty) -> range:
        """ The values of a sort. """
        return range(self.sizes[sort])

    def holds(self, graph: frobenius.Hypergraph,
              assignment: Sequence[int]) -> bool:
        """
        Whether an assignment, a value for each variable, satisfies every
        constraint of an instance: not when it has a value for too many or
        too few variables, or one outside the domain of its sort.

        Parameters:
            graph : The instance.
            assignment : The value of each variable, in order.
        """
        sorts = graph.spider_types
        return len(assignment) == len(sorts) and all(
            value in self.domain(sort)
            for value, sort in zip(assignment, sorts)) and all(
                self.predicates[name](*(assignment[i] for i in scope))
                for name, scope in scopes(graph))

    def solutions(self, graph: frobenius.Hypergraph) -> Iterator[tuple]:
        """
        The assignments that satisfy an instance, by backtracking over its
        variables in order, checking each constraint once its scope is
        assigned: the slow general one, see :meth:`functor`.

        Parameters:
            graph : The instance.

        Example
        -------
        >>> x = frobenius.Ty('x')
        >>> language = Language({x: 3}, {"neq": lambda a, b: a != b})
        >>> triangle = instance(
        ...     [("neq", (0, 1)), ("neq", (1, 2)), ("neq", (0, 2))], 3 * [x])
        >>> len(list(language.solutions(triangle)))
        6
        """
        constraints = scopes(graph)
        checks = [[] for _ in graph.spider_types]
        for name, scope in constraints:
            if scope:
                checks[max(scope)].append((self.predicates[name], scope))
            elif not self.predicates[name]():
                return
        domains = [self.domain(sort) for sort in graph.spider_types]
        assignment: list[int] = []

        def search(variable: int) -> Iterator[tuple]:
            if variable == len(domains):
                yield tuple(assignment)
                return
            for value in domains[variable]:
                assignment.append(value)
                if all(predicate(*(assignment[i] for i in scope))
                       for predicate, scope in checks[variable]):
                    yield from search(variable + 1)
                assignment.pop()
        yield from search(0)

    def array(self, box: cat.Box) -> list:
        """
        The relation of a constraint box as a flat list of booleans, one per
        tuple of values of its sorts, its inputs then its outputs, in
        row-major order, by enumeration.

        Parameters:
            box : A box ``Box(name, dom, cod)``.
        """
        if box not in self.arrays:
            predicate = self.predicates[box.name]
            domains = [self.domain(frobenius.Ty(obj))
                       for obj in (box.dom @ box.cod).inside]
            self.arrays[box] = [
                bool(predicate(*values)) for values in product(*domains)]
        return self.arrays[box]

    def functor(self, dtype: type = bool, **params) -> tensor.Functor:
        """
        The tensor functor of a language: a sort goes to its size and a
        symbol to its relation as an array of ``dtype``, so that an instance
        goes to the projection of its solutions on its outputs over
        ``bool``, and to their number over ``int``: the fast one, see
        :meth:`solutions`.

        Parameters:
            dtype : The datatype of the arrays.
            params : Passed to :class:`tensor.Functor`, e.g. ``contractor``.
        """
        ob_map: dict[cat.Ob, tensor.Dim] = {
            sort: tensor.Dim(size) for sort, size in self.sizes.items()}
        return tensor.Functor(ob_map, self.array, dtype=dtype, **params)

    def propagate(
            self, graph: frobenius.Hypergraph,
            domains: Sequence[set[int]] | None = None,
            rounds: int | None = None) -> list[set[int]]:
        """
        Arc consistency: a set of candidate values for each variable, and in
        each round, for each constraint and each variable of its scope, only
        the values with a support, i.e. an assignment of the scope from the
        candidates that the constraint holds on. Repeated until no set
        shrinks, or for a given number of rounds.

        It removes no value that a solution takes. When the incidence graph
        of the instance, a node per variable and per constraint, is a tree,
        its fixed point is the projection of the solutions on each
        variable :cite:p:`Freuder82`. On any graph, its empty sets are the
        zeros belief propagation finds :cite:p:`DechterMateescu03`.

        Parameters:
            graph : The instance.
            domains : The candidates of each variable, by default every
                value of its sort.
            rounds : The number of rounds, by default until a fixed point.

        Example
        -------
        >>> x = frobenius.Ty('x')
        >>> language = Language({x: 3}, {"lt": lambda a, b: a < b})
        >>> chain = instance([("lt", (0, 1)), ("lt", (1, 2))], 3 * [x])
        >>> language.propagate(chain)
        [{0}, {1}, {2}]
        >>> language.propagate(chain, rounds=1)
        [{0, 1}, {1}, {2}]
        """
        candidates = [set(self.domain(sort)) for sort in graph.spider_types]\
            if domains is None else [set(domain) for domain in domains]
        constraints = [(self.predicates[name], scope)
                       for name, scope in scopes(graph)]
        done, changed = 0, True
        while changed and (rounds is None or done < rounds):
            changed, done = False, done + 1
            for predicate, scope in constraints:
                supports = [
                    values for values in product(
                        *(candidates[i] for i in scope))
                    if predicate(*values)]
                for k, variable in enumerate(scope):
                    kept = {values[k] for values in supports}
                    if kept < candidates[variable]:
                        candidates[variable] &= kept
                        changed = True
        return candidates


def all_different(*values: int) -> bool:
    """ Whether the values are pairwise distinct. """
    return len(set(values)) == len(values)


def exactly_one(*bits: int) -> bool:
    """ Whether exactly one of the bits is set. """
    return sum(bits) == 1


def at_most_one(*bits: int) -> bool:
    """ Whether at most one of the bits is set. """
    return sum(bits) <= 1


def zero_or_two(*bits: int) -> bool:
    """ Whether none or two of the bits are set, i.e. a path goes through
    a cell or it does not. """
    return sum(bits) in (0, 2)


def disjunction(*bits: int) -> bool:
    """ Whether one of the bits is set, a clause on its literals. """
    return any(bits)


def negation(bit: int, other: int) -> bool:
    """ Whether two bits differ, i.e. one is the negation of the other. """
    return bit != other


BIT = frobenius.Ty('bit')
""" The sort of the Boolean variables, of size two. """


def sudoku(n: int) -> tuple[frobenius.Hypergraph, Language]:
    """
    The sudoku of blocks of side ``n``: a variable per cell of the
    ``n ** 2`` by ``n ** 2`` grid, with ``n ** 2`` values, the cells
    numbered row by row, and ``AllDifferent`` on each row, column and
    block.

    Parameters:
        n : The side of a block, 3 for the usual sudoku.

    Example
    -------
    >>> graph, language = sudoku(2)
    >>> len(graph.spider_types), len(graph.boxes)
    (16, 12)
    >>> language.functor(int)(graph)
    Tensor[int]([288], dom=Dim(1), cod=Dim(1))
    """
    side, digit = n * n, frobenius.Ty('digit')
    rows = [tuple(range(side * r, side * (r + 1))) for r in range(side)]
    columns = [tuple(column) for column in zip(*rows)]
    blocks = [tuple(rows[n * i + k][n * j + m]
                    for k in range(n) for m in range(n))
              for i in range(n) for j in range(n)]
    constraints = [("AllDifferent", scope)
                   for scope in rows + columns + blocks]
    return instance(constraints, side * side * [digit]), Language(
        {digit: side}, {"AllDifferent": all_different})


def queens(n: int) -> tuple[frobenius.Hypergraph, Language]:
    """
    The ``n`` queens: a bit per square of the board, numbered row by row,
    saying whether a queen is on it, ``ExactlyOne`` on each row and column
    and ``AtMostOne`` on each diagonal of two squares or more.

    Parameters:
        n : The side of the board.

    Example
    -------
    >>> graph, language = queens(4)
    >>> len(graph.spider_types), len(graph.boxes)
    (16, 18)
    >>> sorted(language.solutions(graph))[0]
    (0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0)
    """
    rows = [tuple(range(n * r, n * (r + 1))) for r in range(n)]
    lines = rows + [tuple(column) for column in zip(*rows)]
    diagonals = [
        tuple(rows[r][c] for r in range(n) for c in range(n)
              if (r - c if anti else r + c) == k)
        for anti in (False, True) for k in range(-n + 1, 2 * n - 1)]
    constraints = [("ExactlyOne", scope) for scope in lines] + [
        ("AtMostOne", scope) for scope in diagonals if len(scope) > 1]
    return instance(constraints, n * n * [BIT]), Language(
        {BIT: 2}, {"ExactlyOne": exactly_one, "AtMostOne": at_most_one})


def cnf(clauses: Sequence[Sequence[int]]
        ) -> tuple[frobenius.Hypergraph, Language]:
    """
    A formula in conjunctive normal form, a clause a list of non-zero
    integers as in DIMACS, ``-i`` the negation of the variable ``i``: a bit
    per variable, numbered from zero, ``Or`` on the literals of each clause,
    and each negated occurrence a fresh bit plugged to its variable through
    ``Not``, the relation of two distinct bits. So ``Or`` is one symmetric
    symbol at every arity, and the solutions are those of the formula, each
    fresh bit being fixed by its variable. The instance numbers the bits as
    they first occur, see :func:`instance`: here ``Not`` first, on the
    second variable and its negation.

    Parameters:
        clauses : The clauses, each a list of literals.

    Example
    -------
    >>> graph, language = cnf([[1, -2], [2, 3]])
    >>> scopes(graph)
    [('Not', (0, 1)), ('Or', (2, 1)), ('Or', (0, 3))]
    >>> language.functor(int)(graph)
    Tensor[int]([4], dom=Dim(1), cod=Dim(1))
    """
    n_variables = max((abs(literal) for clause in clauses
                       for literal in clause), default=0)
    constraints, fresh = [], n_variables
    for clause in clauses:
        scope = []
        for literal in clause:
            if literal > 0:
                scope.append(literal - 1)
                continue
            constraints.append(("Not", (-literal - 1, fresh)))
            scope.append(fresh)
            fresh += 1
        constraints.append(("Or", tuple(scope)))
    return instance(constraints, fresh * [BIT]), Language(
        {BIT: 2}, {"Or": disjunction, "Not": negation})


def maze_diagram(grid: Sequence[str]) -> frobenius.Diagram:
    """
    The diagram of a maze drawn as the maze is: a box per walkable cell, a
    wire per edge between two adjacent walkable cells, i.e. a bit saying
    whether the path takes that edge. A row of the maze is a layer, a
    vertical edge a wire down to the next row and a horizontal edge a cup
    between two neighbours, so that the diagram is planar.

    A cell with neighbours ``up``, ``left``, ``down``, ``right`` walkable
    is ``Box(name, bit ** up, bit ** (left + down + right))``, named after
    its character: ``S`` the start, ``G`` the goal, ``.`` any other.

    Parameters:
        grid : The rows of the maze, ``#`` a wall.

    Example
    -------
    >>> grid = ["S..#.",
    ...         "##.#.",
    ...         ".....",
    ...         ".###.",
    ...         "..G#."]
    >>> maze_diagram(grid).draw(
    ...     wire_labels=False, figsize=(5, 5),
    ...     doctest='docs/_static/csp/maze.svg')

    .. image:: /_static/csp/maze.svg
        :align: center
    """
    def walkable(r, c):
        return 0 <= r < len(grid) and 0 <= c < len(grid[r])\
            and grid[r][c] != '#'
    diagram = frobenius.Id()
    for r, row in enumerate(grid):
        boxes, plumbing = frobenius.Id(), frobenius.Id()
        for c, char in enumerate(row):
            if char == '#':
                continue
            up, left, down, right = (walkable(r + i, c + j) for i, j in (
                (-1, 0), (0, -1), (1, 0), (0, 1)))
            boxes @= frobenius.Box(
                char, BIT ** up, BIT ** (left + down + right))
            plumbing @= (frobenius.Cup(BIT, BIT) if left else frobenius.Id())\
                @ BIT ** down
        diagram >>= (boxes >> plumbing).foliation()
    return diagram


def maze(grid: Sequence[str]) -> tuple[frobenius.Hypergraph, Language]:
    """
    The maze of :func:`maze_diagram` as an instance: a bit per edge, and on
    the edges of each cell ``ExactlyOne`` at the start and the goal,
    ``ZeroOrTwo`` anywhere else. When the walkable cells form a tree, its
    one solution is the path from the start to the goal. With a cycle the
    constraints also take the path with any disjoint loops beside it.

    Parameters:
        grid : The rows of the maze, ``#`` a wall.

    Example
    -------
    >>> graph, language = maze(["S..#.",
    ...                         "##.#.",
    ...                         ".....",
    ...                         ".###.",
    ...                         "..G#."])
    >>> len(graph.spider_types), len(graph.boxes)
    (16, 17)
    >>> language.functor(int)(graph)
    Tensor[int]([1], dom=Dim(1), cod=Dim(1))
    """
    graph = frobenius.Hypergraph.from_diagram(maze_diagram(grid))
    return graph, Language({BIT: 2}, {
        'S': exactly_one, 'G': exactly_one, '.': zero_or_two})
