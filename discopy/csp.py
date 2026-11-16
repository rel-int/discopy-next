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
    instance itself.

    A box may also have inputs, bent into outputs as a hypergraph category
    allows: its scope is the variables on its inputs, then on its outputs,
    so that a diagram drawn in the plane, e.g. a maze with a wire per edge,
    reads as the instance it draws.

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
    return [(box.name, dom_wires + cod_wires)
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
        constraint of an instance.

        Parameters:
            graph : The instance.
            assignment : The value of each variable, in order.
        """
        return all(
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
