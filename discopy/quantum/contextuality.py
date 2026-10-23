# -*- coding: utf-8 -*-

"""
The contextual fraction of empirical models, a measure of contextuality
computed by linear programming, following Abramsky, Barbosa and Mansfield
:cite:`AbramskyEtAl17`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Scenario
    Table
    Model
    Inequality

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        maximise

Note
----
A measurement scenario :math:`\\langle X, \\mathcal{M}, O \\rangle` is a set
of measurements :math:`X` covered by contexts :math:`C \\in \\mathcal{M}`,
the sets of measurements that can be performed together, each with outcomes
:math:`O`. An empirical model :math:`e` is a probability distribution
:math:`e_C` on the local assignments :math:`s \\in O^C` of each context,
flattened into a vector :math:`v^e` indexed by the local sections
:math:`\\langle C, s \\rangle`. It is non-contextual when it is the marginal
of a distribution :math:`d` on the global assignments :math:`g \\in O^X`, i.e.
when :math:`M d = v^e` for the incidence matrix with
:math:`M[\\langle C, s \\rangle, g] = 1` if :math:`g|_C = s` and :math:`0`
otherwise.

The non-contextual fraction :math:`\\mathsf{NCF}(e)` is the largest weight
:math:`1 \\cdot b` of a subdistribution :math:`b \\geq 0` on global
assignments with :math:`M b \\leq v^e`, the contextual fraction is
:math:`\\mathsf{CF}(e) = 1 - \\mathsf{NCF}(e)`: it is :math:`0` for
non-contextual models and :math:`1` for strongly contextual ones. An optimal
:math:`b` splits the model as
:math:`e = \\mathsf{NCF}(e) e^{NC} + \\mathsf{CF}(e) e^{SC}`, a non-contextual
model plus a strongly contextual one. The dual program yields a Bell
inequality whose normalised violation by :math:`e` is exactly
:math:`\\mathsf{CF}(e)`, saturated by :math:`e^{NC}` and maximally violated by
:math:`e^{SC}`.

Quantum empirical models arise from local measurements on a state: the
:class:`discopy.quantum.channel.Functor` sends the circuit
``state >> m_1 @ ... @ m_n`` of each context to the distribution
``F(state) >> F(m_1) @ ... @ F(m_n)``.

Example
-------
The Popescu-Rohrlich box is strongly contextual, the Bell-CHSH model of
measurements at angles :math:`0` and :math:`\\pi / 3` on each qubit of the
Bell state is a quarter contextual.

>>> from math import pi
>>> from discopy.quantum import Ket, H, CX, Rz, Measure, qubit
>>> PR = Model(Scenario.bell(parties=2), [[1/2, 0, 0, 1/2]] * 3 + [
...     [0, 1/2, 1/2, 0]])
>>> PR.scenario.contexts
(('a1', 'b1'), ('a1', 'b2'), ('a2', 'b1'), ('a2', 'b2'))
>>> round(PR.contextual_fraction(), 6)  # doctest: +EXTRA
1.0
>>> phi_plus = Ket(0, 0) >> H @ qubit >> CX
>>> measure = lambda phi: Rz(-phi / (2 * pi)) >> H >> Measure()
>>> CHSH = Model.from_circuits(phi_plus, 2 * [[measure(0), measure(pi / 3)]])
>>> print(*[[round(p, 3) for p in row] for row in CHSH.table], sep="\\n")
[0.5, 0.0, 0.0, 0.5]
[0.375, 0.125, 0.125, 0.375]
[0.375, 0.125, 0.125, 0.375]
[0.125, 0.375, 0.375, 0.125]
>>> round(CHSH.contextual_fraction(), 6)
0.25
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product

import numpy as np

from discopy.quantum.channel import Channel
from discopy.quantum.circuit import Circuit


def maximise(objective: np.ndarray, matrix: np.ndarray, vector: np.ndarray,
             bounds: tuple) -> np.ndarray:
    """
    An optimal solution :math:`x` of the linear program maximising
    :math:`\\mathrm{objective} \\cdot x` subject to
    :math:`\\mathrm{matrix} \\, x \\leq \\mathrm{vector}` and ``bounds`` on
    each component of :math:`x`, raising ``ValueError`` when there is none.
    """
    # pylint: disable=import-outside-toplevel
    # scipy is an optional dependency, in the quantum extra.
    from scipy.optimize import linprog
    result = linprog(-objective, A_ub=matrix, b_ub=vector, bounds=bounds)
    if not result.success:
        raise ValueError(result.message)
    return result.x


class Scenario:
    """
    A measurement scenario is a tuple of contexts, each a tuple of
    measurements, with a number of outcomes for each measurement.

    Parameters:
        contexts : The sets of measurements that can be performed together.
        outcomes : The number of outcomes of each measurement.

    Example
    -------
    >>> scenario = Scenario([('x', 'y'), ('y', 'z')])
    >>> scenario.measurements
    ('x', 'y', 'z')
    >>> scenario.incidence()
    array([[1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1],
           [1, 0, 0, 0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 0, 0, 1]])
    """
    def __init__(self, contexts: Sequence[Sequence], outcomes: int = 2):
        self.contexts, self.outcomes = tuple(map(tuple, contexts)), outcomes

    def __eq__(self, other):
        return isinstance(other, Scenario) and (
            self.contexts, self.outcomes) == (other.contexts, other.outcomes)

    def __hash__(self):
        return hash((self.contexts, self.outcomes))

    def __repr__(self):
        return f"Scenario({self.contexts}, outcomes={self.outcomes})"

    @property
    def measurements(self) -> tuple:
        """ The measurements :math:`X`, in order of appearance. """
        return tuple(dict.fromkeys(
            x for context in self.contexts for x in context))

    @property
    def assignments(self) -> list[tuple[int, ...]]:
        """ The global assignments :math:`O^X`, in lexicographic order. """
        return list(product(
            range(self.outcomes), repeat=len(self.measurements)))

    def incidence(self) -> np.ndarray:
        """
        The incidence matrix :math:`M`, with a column for each global
        assignment :math:`g \\in O^X` and a row for each local section
        :math:`\\langle C, s \\rangle`, listing each context's assignments
        :math:`s \\in O^C` in lexicographic order.
        """
        measurements, assignments = self.measurements, np.array(
            self.assignments, dtype=int)
        result = np.zeros((
            sum(self.outcomes ** len(context) for context in self.contexts),
            len(assignments)), dtype=int)
        offset = 0
        for context in self.contexts:
            indices = [measurements.index(x) for x in context]
            powers = self.outcomes ** np.arange(len(context))[::-1]
            sections = offset + assignments[:, indices] @ powers
            result[sections, np.arange(len(assignments))] = 1
            offset += self.outcomes ** len(context)
        return result

    def rows(self, vector: Sequence[float]) -> list[np.ndarray]:
        """ Split a vector indexed by local sections into rows. """
        lengths = [self.outcomes ** len(context) for context in self.contexts]
        return np.split(np.asarray(vector), np.cumsum(lengths)[:-1])

    def relabel(self, mapping: dict) -> Scenario:
        """ Rename the measurements, the ones not in ``mapping`` stay. """
        if len({mapping.get(x, x) for x in self.measurements}) < len(
                self.measurements):
            raise ValueError(f"{mapping} identifies some measurements.")
        return Scenario([[mapping.get(x, x) for x in context]
                         for context in self.contexts], self.outcomes)

    def choice(self, other: Scenario) -> Scenario:
        """
        The scenario with the contexts of either scenario, called with
        ``&``. Their measurements must be disjoint.
        """
        self.assert_disjoint(other)
        return Scenario(self.contexts + other.contexts, self.outcomes)

    def tensor(self, other: Scenario) -> Scenario:
        """
        The scenario with a context for each pair of contexts, called with
        ``@``. Their measurements must be disjoint.
        """
        self.assert_disjoint(other)
        return Scenario([
            left + right for left in self.contexts for right in other.contexts
        ], self.outcomes)

    def assert_disjoint(self, other: Scenario):
        """
        Raise ``ValueError`` unless two scenarios have the same outcomes and
        disjoint measurements.
        """
        if self.outcomes != other.outcomes:
            raise ValueError(
                f"Outcomes {self.outcomes} and {other.outcomes} differ.")
        if shared := set(self.measurements) & set(other.measurements):
            raise ValueError(
                f"Measurements {sorted(shared)} are shared, relabel them.")

    __and__, __matmul__ = choice, tensor

    @classmethod
    def bell(cls, parties: int, settings: int = 2, outcomes: int = 2
             ) -> Scenario:
        """
        The Bell scenario where each party, named ``a``, ``b``, ``c``, etc.,
        performs one of its measurements, numbered from ``1``.

        Parameters:
            parties : The number of parties.
            settings : The number of measurements of each party.
            outcomes : The number of outcomes of each measurement.
        """
        return cls(product(*(  # ty: ignore[invalid-argument-type]
            [f"{chr(ord('a') + i)}{j + 1}" for j in range(settings)]
            for i in range(parties))), outcomes)


class Table:
    """
    A table assigns a number to each local section
    :math:`\\langle C, s \\rangle` of a scenario, with a row for each context.

    Parameters:
        scenario : The measurement scenario.
        table : For each context :math:`C`, a row indexed by :math:`O^C`.
    """
    def __init__(self, scenario: Scenario, table: Sequence[Sequence[float]]):
        table = tuple(tuple(map(float, row)) for row in table)
        if [len(row) for row in table] != [
                scenario.outcomes ** len(context)
                for context in scenario.contexts]:
            raise ValueError(f"The table does not fit {scenario}.")
        self.scenario, self.table = scenario, table

    def __eq__(self, other):
        return type(self) is type(other) and (self.scenario, self.table) == (
            other.scenario, other.table)

    def __hash__(self):
        return hash((self.scenario, self.table))

    def __repr__(self):
        return f"{type(self).__name__}({self.scenario!r}, {self.table})"

    @property
    def vector(self) -> np.ndarray:
        """ The table flattened into a vector indexed by local sections. """
        return np.array([x for row in self.table for x in row])

    def is_close(self, other: Table, rtol=1e-5, atol=1e-8) -> bool:
        """ Whether two tables on the same scenario are numerically close. """
        return self.scenario == other.scenario and np.allclose(
            self.vector, other.vector, rtol=rtol, atol=atol)


class Model(Table):
    """
    An empirical model is a table of probability distributions, one for each
    context, with compatible marginals.

    Parameters:
        scenario : The measurement scenario.
        table : For each context :math:`C`, a distribution on :math:`O^C`.

    Example
    -------
    Mixing the Popescu-Rohrlich box with white noise, the model is contextual
    only when the box has weight more than a half.

    >>> PR = Model(Scenario.bell(2), [[1/2, 0, 0, 1/2]] * 3 + [
    ...     [0, 1/2, 1/2, 0]])
    >>> noise = Model(Scenario.bell(2), [[1/4] * 4] * 4)
    >>> for weight in [1/4, 1/2, 3/4]:  # doctest: +EXTRA
    ...     print(round(PR.mix(noise, weight).contextual_fraction(), 6))
    0.0
    0.0
    0.5

    The table is not checked until a linear program is solved: then every row
    must be a distribution and the marginals of any two contexts must agree on
    their shared measurements.

    >>> signalling = Model(Scenario.bell(2), [[1, 0, 0, 0], [0, 0, 1, 0]]
    ...     + [[1/4] * 4] * 2)
    >>> signalling.is_compatible()
    False
    >>> signalling.contextual_fraction()
    Traceback (most recent call last):
    ...
    ValueError: The table is not an empirical model.
    """
    def marginal(self, i: int, measurements: Sequence) -> np.ndarray:
        """
        The marginal :math:`e_C|_U` of the distribution of the ``i``-th
        context :math:`C` on some of its ``measurements`` :math:`U`, indexed
        by :math:`O^U` in lexicographic order.
        """
        context, outcomes = self.scenario.contexts[i], self.scenario.outcomes
        axes = [context.index(x) for x in measurements]
        marginal = np.reshape(self.table[i], len(context) * (outcomes, )).sum(
            axis=tuple(j for j in range(len(context)) if j not in axes))
        return np.transpose(marginal, np.argsort(np.argsort(axes))).flatten()

    def is_compatible(self, atol: float = 1e-9) -> bool:
        """
        Whether the table is an empirical model up to ``atol``: its rows are
        probability distributions and the marginals of any two contexts agree
        on their shared measurements.
        """
        contexts = self.scenario.contexts
        return bool((self.vector >= -atol).all()) and all(
            abs(sum(row) - 1) <= atol for row in self.table) and all(
            np.allclose(
                self.marginal(i, shared), self.marginal(j, shared),
                rtol=0, atol=atol)
            for i, j in combinations(range(len(contexts)), 2)
            for shared in [[x for x in contexts[i] if x in contexts[j]]])

    def assert_compatible(self):
        """ Raise ``ValueError`` unless the table is an empirical model. """
        if not self.is_compatible():
            raise ValueError("The table is not an empirical model.")

    def global_subdistribution(self) -> np.ndarray:
        """
        A subdistribution :math:`b` on global assignments of maximal weight
        with :math:`M b \\leq v^e`, an optimal solution of the linear program
        (3) of :cite:`AbramskyEtAl17`.
        """
        self.assert_compatible()
        incidence = self.scenario.incidence()
        return maximise(
            np.ones(incidence.shape[1]), incidence, self.vector, (0, None))

    def non_contextual_fraction(self) -> float:
        """ The non-contextual fraction :math:`\\mathsf{NCF}(e)`. """
        return float(self.global_subdistribution().sum())

    def contextual_fraction(self) -> float:
        """ The contextual fraction :math:`\\mathsf{CF}(e)`. """
        return 1 - self.non_contextual_fraction()

    def decompose(self) -> tuple[float, Model | None, Model | None]:
        """
        The decomposition :math:`e = \\mathsf{NCF}(e) e^{NC}
        + \\mathsf{CF}(e) e^{SC}` into a non-contextual and a strongly
        contextual model, the one of weight zero up to :math:`10^{-9}` being
        ``None``.

        Example
        -------
        >>> CHSH = Model(Scenario.bell(2), [[1/2, 0, 0, 1/2]] + 2 * [
        ...     [3/8, 1/8, 1/8, 3/8]] + [[1/8, 3/8, 3/8, 1/8]])
        >>> weight, NC, SC = CHSH.decompose()  # doctest: +EXTRA
        >>> round(weight, 6)
        0.75
        >>> round(NC.contextual_fraction(), 6), round(
        ...     SC.contextual_fraction(), 6)
        (0.0, 1.0)
        >>> assert NC.mix(SC, weight).is_close(CHSH)
        """
        subdistribution = self.global_subdistribution()
        weight = float(subdistribution.sum())
        local = self.scenario.incidence() @ subdistribution
        non_contextual = None if np.isclose(
            weight, 0, rtol=0, atol=1e-9) else Model(
                self.scenario, self.scenario.rows(
                    local / weight))  # ty: ignore[invalid-argument-type]
        strongly_contextual = None if np.isclose(
            weight, 1, rtol=0, atol=1e-9) else Model(
            self.scenario,
            self.scenario.rows(  # ty: ignore[invalid-argument-type]
                (self.vector - local) / (1 - weight)))
        return weight, non_contextual, strongly_contextual

    def bell_inequality(self) -> Inequality:
        """
        A Bell inequality :math:`a \\cdot v \\leq 0` whose normalised
        violation by the model is its contextual fraction, an optimal
        solution of the linear program (7) of :cite:`AbramskyEtAl17`.

        Example
        -------
        >>> CHSH = Model(Scenario.bell(2), [[1/2, 0, 0, 1/2]] + 2 * [
        ...     [3/8, 1/8, 1/8, 3/8]] + [[1/8, 3/8, 3/8, 1/8]])
        >>> inequality = CHSH.bell_inequality()  # doctest: +EXTRA
        >>> round(inequality.non_contextual_bound, 6) <= 0
        True
        >>> round(inequality.violation(CHSH), 6)
        0.25
        """
        self.assert_compatible()
        incidence = self.scenario.incidence()
        coefficients = maximise(
            self.vector, incidence.T, np.zeros(incidence.shape[1]),
            (None, 1 / len(self.scenario.contexts)))
        rows = self.scenario.rows(
            coefficients)  # ty: ignore[invalid-argument-type]
        return Inequality(
            self.scenario, rows)  # ty: ignore[invalid-argument-type]

    def mix(self, other: Model, weight: float) -> Model:
        """
        The mixture :math:`\\lambda e + (1 - \\lambda) e'` of two models on
        the same scenario.

        Parameters:
            other : The model :math:`e'`.
            weight : The weight :math:`\\lambda \\in [0, 1]` of ``self``.
        """
        if self.scenario != other.scenario:
            raise ValueError(f"{self.scenario} != {other.scenario}.")
        if not 0 <= weight <= 1:
            raise ValueError(f"{weight} is not in the unit interval.")
        return Model(
            self.scenario,
            self.scenario.rows(  # ty: ignore[invalid-argument-type]
                weight * self.vector + (1 - weight) * other.vector))

    def relabel(self, mapping: dict) -> Model:
        """ Rename the measurements, the ones not in ``mapping`` stay. """
        return Model(self.scenario.relabel(mapping), self.table)

    def choice(self, other: Model) -> Model:
        """
        The controlled choice :math:`e \\& e'` of two models, called with
        ``&``, where either one is measured.
        """
        return Model(self.scenario & other.scenario, self.table + other.table)

    def tensor(self, other: Model) -> Model:
        """
        The product :math:`e \\otimes e'` of two models, called with ``@``,
        where both are measured independently.
        """
        table = [np.outer(left, right).flatten()
                 for left in self.table for right in other.table]
        return Model(
            self.scenario @ other.scenario,
            table)  # ty: ignore[invalid-argument-type]

    __and__, __matmul__ = choice, tensor

    @classmethod
    def from_circuits(cls, state: Circuit, measurements: list[list[Circuit]]
                      ) -> Model:
        """
        The empirical model on a Bell scenario where each party measures one
        qubit of a ``state``.

        Parameters:
            state : A circuit from no qubits to ``n`` qubits.
            measurements : For each of the ``n`` parties, its measurements as
                circuits from ``qubit`` to ``bit``.
        """
        density = state.eval(mixed=True)
        channels = [[m.eval(mixed=True) for m in party]
                    for party in measurements]
        if len({len(party) for party in channels}) != 1 or len(
                {channel.cod for party in channels for channel in party}) != 1:
            raise ValueError("Every party needs as many measurements, "
                             "all with the same outcomes.")
        scenario = Scenario.bell(
            len(measurements), len(measurements[0]),
            channels[0][0].cod.classical.inside[0])
        return cls(scenario, [
            (density >> Channel.id().tensor(*(
                party[j] for party, j in zip(channels, choice)))
             ).array.real.flatten()
            for choice in product(range(len(measurements[0])),
                                  repeat=len(measurements))])


class Inequality(Table):
    """
    An inequality :math:`a \\cdot v^e \\leq R` on the empirical models of a
    scenario, given by a table of coefficients :math:`a` and a bound
    :math:`R`.

    Parameters:
        scenario : The measurement scenario.
        table : For each context :math:`C`, coefficients indexed by
            :math:`O^C`.
        bound : The bound :math:`R`.

    Example
    -------
    The CHSH inequality, as a sum of the probabilities of the events where
    the outcomes are correlated, or anti-correlated in the last context.

    >>> CHSH = Inequality(Scenario.bell(2), [[1, 0, 0, 1]] * 3 + [
    ...     [0, 1, 1, 0]], bound=3)
    >>> CHSH.non_contextual_bound, CHSH.algebraic_bound
    (3.0, 4.0)
    """
    def __init__(self, scenario: Scenario, table: Sequence[Sequence[float]],
                 bound: float = 0):
        super().__init__(scenario, table)
        self.bound = float(bound)

    def __eq__(self, other):
        return super().__eq__(other) and self.bound == other.bound

    def __hash__(self):
        return hash((self.scenario, self.table, self.bound))

    def __repr__(self):
        return f"Inequality({self.scenario!r}, {self.table}, " \
               f"bound={self.bound})"

    def __call__(self, model: Model) -> float:
        """ The value :math:`a \\cdot v^e` of the inequality at a model. """
        if self.scenario != model.scenario:
            raise ValueError(f"{self.scenario} != {model.scenario}.")
        return float(self.vector @ model.vector)

    @property
    def non_contextual_bound(self) -> float:
        """
        The largest value at a non-contextual model, i.e. at a deterministic
        model given by a global assignment: the inequality is a Bell
        inequality when this is at most its bound.
        """
        return float(max(self.vector @ self.scenario.incidence()))

    @property
    def algebraic_bound(self) -> float:
        """
        The largest value :math:`\\Vert a \\Vert` at any model, the sum over
        contexts of the largest coefficient.
        """
        return float(sum(max(row) for row in self.table))

    def violation(self, model: Model) -> float:
        """
        The normalised violation
        :math:`\\max\\{0, a \\cdot v^e - R\\} / (\\Vert a \\Vert - R)`
        of the inequality by a model.
        """
        if self.algebraic_bound <= self.bound:
            raise ValueError(f"{self} cannot be violated.")
        return max(0., self(model) - self.bound) / (
            self.algebraic_bound - self.bound)
