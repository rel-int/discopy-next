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

        solve
        maximise
        sensitivity
        ascend
        attempt
        derivative

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
``F(state) >> F(m_1) @ ... @ F(m_n)``. They also arise from an optyx diagram
with a classical input choosing the measurement of each party and a classical
output giving its outcome, see :meth:`Model.from_diagram`.

The contextual fraction is differentiable through the linear program: its
value is piecewise linear in :math:`v^e` and its gradient, wherever it is
differentiable, is the solution of the dual program, which the solver
returns as the shadow prices of the constraints :math:`M b \\leq v^e`, see
:meth:`Model.gradient`. The gradient is zero on the non-contextual polytope
and a subgradient on its boundary, where the dual solution is not unique.
Composed with the derivatives of a parametrised model, this is the gradient
ascent of :func:`ascend`, finding the parameters of a setup that maximise
its contextual fraction from a contextual starting point.

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

from collections.abc import Callable, Sequence
from itertools import combinations, product

import numpy as np

from discopy.quantum.channel import Channel
from discopy.quantum.circuit import Circuit


def solve(objective: np.ndarray, matrix: np.ndarray, vector: np.ndarray,
          bounds: tuple):
    """
    The result of solving the linear program maximising
    :math:`\\mathrm{objective} \\cdot x` subject to
    :math:`\\mathrm{matrix} \\, x \\leq \\mathrm{vector}` and ``bounds`` on
    each component of :math:`x`, raising ``ValueError`` when there is no
    solution.
    """
    # pylint: disable=import-outside-toplevel
    # scipy is an optional dependency, in the quantum extra.
    from scipy.optimize import linprog
    result = linprog(-objective, A_ub=matrix, b_ub=vector, bounds=bounds)
    if not result.success:
        raise ValueError(result.message)
    return result


def maximise(objective: np.ndarray, matrix: np.ndarray, vector: np.ndarray,
             bounds: tuple) -> np.ndarray:
    """ An optimal solution :math:`x` of the linear program of :func:`solve`.
    """
    return solve(objective, matrix, vector, bounds).x


def sensitivity(objective: np.ndarray, matrix: np.ndarray,
                vector: np.ndarray, bounds: tuple) -> np.ndarray:
    """
    The gradient of the value of the linear program of :func:`solve` with
    respect to ``vector``: the value is piecewise linear in it and its
    gradient is the solution of the dual program, which the solver returns
    as the shadow prices of the constraints, a subgradient where the dual
    solution is not unique.
    """
    return -solve(objective, matrix, vector, bounds).ineqlin.marginals


def ascend(model: Callable[[np.ndarray], Model],
           params: Sequence[float] | np.ndarray,
           steps: int = 10, learning_rate: float = .05,
           difference: float = 1e-3) -> tuple[np.ndarray, Model]:
    """
    Gradient ascent on the contextual fraction of a parametrised model,
    returning the parameters and the model it stops at.

    Parameters:
        model : A function from an array of parameters to a model, raising
            ``ValueError`` outside the parameters it is defined on.
        params : The starting parameters.
        steps : The number of gradients computed.
        learning_rate : The initial step size, doubled after a step that
            improves the fraction and halved, up to eight times, after one
            that does not or that leaves the empirical models.
        difference : The step of the finite differences approximating the
            derivatives of the table with respect to the parameters.

    The gradient of the fraction with respect to the table is
    :meth:`Model.gradient`, exact, and the chain rule contracts it with the
    derivatives of the table with respect to the parameters, the only
    approximation. The fraction is piecewise linear in the table, so a fixed
    step overshoots its kinks, which the backtracking is for; it is zero on
    the non-contextual polytope, so the ascent only moves from a contextual
    starting point.

    Example
    -------
    Mixing the Popescu-Rohrlich box with white noise, the ascent on the
    weight of the box from a contextual start approaches the box itself:
    the mixture is not defined beyond a weight of one, so every step past
    it is halved until it lands below.

    >>> PR = Model(Scenario.bell(2), [[1/2, 0, 0, 1/2]] * 3 + [
    ...     [0, 1/2, 1/2, 0]])
    >>> noise = Model(Scenario.bell(2), [[1/4] * 4] * 4)
    >>> isotropic = lambda params: PR.mix(noise, params[0])
    >>> weight, box = ascend(isotropic, [.75])  # doctest: +EXTRA
    >>> assert .99 < weight[0] <= 1 and box.contextual_fraction() > .98
    """
    params = np.array(params, dtype=float)
    current = model(params)
    for _ in range(steps):
        gradient = current.gradient().vector
        jacobian = np.array([
            derivative(model, params, current, difference * direction)
            for direction in np.eye(len(params))])
        direction = jacobian @ gradient
        for _ in range(8):
            candidate = attempt(model, params + learning_rate * direction)
            if candidate is not None and candidate.contextual_fraction()\
                    > current.contextual_fraction():
                params, current = params + learning_rate * direction, candidate
                learning_rate = 2 * learning_rate
                break
            learning_rate = learning_rate / 2
    return params, current


def attempt(model: Callable[[np.ndarray], Model], params: np.ndarray
            ) -> Model | None:
    """
    The model at some parameters, or ``None`` where the setup raises
    ``ValueError`` or its table is not an empirical model, which
    :func:`ascend` steps over rather than stopping at.
    """
    try:
        candidate = model(params)
        candidate.assert_compatible()
    except ValueError:
        return None
    return candidate


def derivative(model: Callable[[np.ndarray], Model], params: np.ndarray,
               current: Model, step: np.ndarray) -> np.ndarray:
    """
    The derivative of the table of a parametrised model along a step, by a
    forward difference from the ``current`` model at ``params``, or a
    backward one where the forward step leaves the setup.
    """
    forward = attempt(model, params + step)
    if forward is not None:
        return (forward.vector - current.vector) / np.linalg.norm(step)
    backward = model(params - step)
    return (current.vector - backward.vector) / np.linalg.norm(step)


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
    def bell(cls, parties: int, settings: int | Sequence[int] = 2,
             outcomes: int = 2) -> Scenario:
        """
        The Bell scenario where each party, named ``a``, ``b``, ``c``, etc.,
        performs one of its measurements, numbered from ``1``.

        Parameters:
            parties : The number of parties.
            settings : The number of measurements of each party, or one
                number for every party.
            outcomes : The number of outcomes of each measurement.

        Example
        -------
        >>> Scenario.bell(2, settings=(3, 1)).contexts
        (('a1', 'b1'), ('a2', 'b1'), ('a3', 'b1'))
        """
        settings = parties * (settings, ) if isinstance(settings, int)\
            else tuple(settings)
        if len(settings) != parties:
            raise ValueError(f"{settings} are not the settings of {parties} "
                             "parties.")
        return cls(product(*(  # ty: ignore[invalid-argument-type]
            [f"{chr(ord('a') + i)}{j + 1}" for j in range(n)]
            for i, n in enumerate(settings))), outcomes)


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

    def gradient(self) -> Table:
        """
        The gradient of the contextual fraction with respect to the table,
        the :func:`sensitivity` of the linear program of
        :meth:`global_subdistribution` negated: a subgradient on the
        boundary of the non-contextual polytope, and constant on each
        context inside it, where the fraction is zero.

        Example
        -------
        The gradient sums to the fraction minus one against the model, and
        to at most minus one against any deterministic model.

        >>> CHSH = Model(Scenario.bell(2), [[1/2, 0, 0, 1/2]] + 2 * [
        ...     [3/8, 1/8, 1/8, 3/8]] + [[1/8, 3/8, 3/8, 1/8]])
        >>> gradient = CHSH.gradient()  # doctest: +EXTRA
        >>> round(float(gradient.vector @ CHSH.vector), 6)
        -0.75
        >>> assert all(gradient.vector @ Model.deterministic(
        ...     CHSH.scenario, g).vector <= -1 + 1e-9
        ...     for g in CHSH.scenario.assignments)
        """
        self.assert_compatible()
        incidence = self.scenario.incidence()
        return Table(self.scenario, self.scenario.rows(-sensitivity(
            np.ones(incidence.shape[1]), incidence, self.vector, (0, None))))

    def normalise(self, atol: float = 1e-9) -> Model:
        """
        The model conditioned on the success of a postselection, each row
        divided by its sum, which must be the same in every context up to
        ``atol``, raising ``ValueError`` otherwise or when the result is
        not an empirical model, e.g. when the postselection signals.

        Example
        -------
        >>> half = Model(Scenario.bell(2), [[1/4, 0, 0, 1/4]] * 3 + [
        ...     [0, 1/4, 1/4, 0]])
        >>> half.normalise().table[0]
        (0.5, 0.0, 0.0, 0.5)
        """
        sums = [sum(row) for row in self.table]
        if min(sums) <= 0 or max(sums) - min(sums) > atol:
            raise ValueError(
                f"The rows of {self} do not sum to the same probability.")
        result = Model(self.scenario, [
            [x / total for x in row] for row, total in zip(self.table, sums)])
        result.assert_compatible()
        return result

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

    @classmethod
    def deterministic(cls, scenario: Scenario, assignment: Sequence[int]
                      ) -> Model:
        """
        The non-contextual model where each measurement has the outcome of a
        global ``assignment``, listed in the order of the measurements.

        Example
        -------
        >>> Model.deterministic(Scenario([('x', 'y')]), (0, 1)).table
        ((0.0, 1.0, 0.0, 0.0),)
        """
        index = scenario.assignments.index(tuple(assignment))
        return cls(scenario, scenario.rows(scenario.incidence()[:, index]))

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

    @classmethod
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # The parties, their settings, outcomes and outputs, and the backend.
    def from_diagram(cls, diagram, settings: int | Sequence[int] = 2,
                     outcomes: int | None = None, backend=None,
                     outputs: int | Sequence[int] = 1) -> Model:
        """
        The empirical model on a Bell scenario of an optyx diagram with one
        classical input and some classical outputs for each party, the input
        choosing the measurement of the party and the outputs its outcome,
        evaluated by :func:`discopy.optyx.contextuality.distributions`.

        Parameters:
            diagram : A :class:`discopy.optyx.channel.Diagram` with classical
                domain and codomain.
            settings : The number of measurements of each party, or one
                number for every party.
            outcomes : The number of outcomes of each output wire, by default
                one more than the largest outcome of non-zero probability.
            backend : The backend evaluating the diagram.
            outputs : The number of output wires of each party, or one
                number for every party, the outcome of a party being the
                tuple of the values of its wires in lexicographic order, so
                that a measurement has ``outcomes ** outputs`` outcomes.

        The probabilities are not normalised: a diagram that is not a channel,
        e.g. a postselected one, does not give an empirical model unless it
        is conditioned on its success with :meth:`normalise`.
        """
        # pylint: disable=import-outside-toplevel
        # discopy.optyx is an optional dependency, in the optyx extra.
        from discopy.optyx.contextuality import distributions
        parties = len(diagram.dom.inside)
        settings = parties * (settings, ) if isinstance(settings, int)\
            else tuple(settings)
        outputs = parties * (outputs, ) if isinstance(outputs, int)\
            else tuple(outputs)
        if len(outputs) != parties or len(set(outputs)) > 1:
            raise ValueError(f"The {parties} parties need the same number "
                             f"of outputs, got {outputs}.")
        rows = distributions(diagram, settings, backend, outputs)
        if outcomes is None:
            outcomes = 1 + max(
                max(section, default=0) for row in rows for section in row)
        wires = sum(outputs)
        table = np.zeros((len(rows), outcomes ** wires))
        for row, distribution in zip(table, rows):
            for section, probability in distribution.items():
                if max(section, default=0) >= outcomes:
                    raise ValueError(
                        f"The outcomes {section} are not in {outcomes}.")
                row[np.ravel_multi_index(
                    section, wires * (outcomes, ))] = probability
        return cls(Scenario.bell(
            parties, settings, outcomes ** outputs[0]), table)


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
        of the inequality by a model, zero when :math:`a \\cdot v^e \\leq R`
        as for the trivial inequality, with :math:`\\Vert a \\Vert = R`,
        that is the Bell inequality of a non-contextual model.

        Example
        -------
        >>> local = Model.deterministic(Scenario.bell(2), (0, 0, 0, 0))
        >>> local.bell_inequality().violation(local)  # doctest: +EXTRA
        0.0
        """
        excess = self(model) - self.bound
        room = self.algebraic_bound - self.bound
        atol = 1e-9 * max(1., float(abs(self.vector).max(initial=0)))
        if excess <= atol:
            return 0.
        if room <= atol:
            raise ValueError(f"{model} exceeds the algebraic bound of {self}.")
        return excess / room
