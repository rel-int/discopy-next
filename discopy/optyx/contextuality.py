"""
Overview
--------

The distributions of the outcomes of a diagram for each choice of settings,
the empirical model :meth:`discopy.quantum.contextuality.Model.from_diagram`
computes the contextual fraction of, and the message-passing scenario of a
:class:`discopy.optyx.interaction.CMap`, where the nodes measure what they
predict after exchanging messages.

Functions
---------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    distributions
    local_measurement
    dual_rail_measurement

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    MessagePassingScenario

Example
-------
A Bell test at the Tsirelson bound, with a classical input for the setting
and a classical output for the outcome of each party, the setting
controlling a phase on its qubit before it is measured:

>>> from discopy.optyx import bit, qubit, Diagram
>>> from discopy.optyx.qubits import Z, H, Scalar, Measure
>>> from discopy.optyx.classical import BitControlledGate
>>> from discopy.quantum.contextuality import Model
>>> alice = BitControlledGate(Z(1, 1, .25)) >> H() >> Measure(1)
>>> bob = Diagram.swap(qubit, bit) >> bit @ Z(1, 1, -.125)\\
...     >> BitControlledGate(Z(1, 1, .25)) >> H() >> Measure(1)
>>> bell_test = bit @ Scalar(.5 ** .5) @ Z(0, 2) @ bit >> alice @ bob
>>> tsirelson = Model.from_diagram(bell_test)
>>> round(tsirelson.contextual_fraction(), 6)  # sqrt(2) - 1
0.414214

The measurements maximising the contextual fraction of a setup are found by
:func:`discopy.quantum.contextuality.ascend`. Here the settings of each
party are two free phases, and from a contextual starting point the ascent
reaches the Tsirelson bound, each party's settings a quarter turn apart:

>>> from discopy.quantum.contextuality import ascend
>>> def bell_test(params):
...     a0, a1, b0, b1 = params
...     alice = bit @ Z(1, 1, a0) >> BitControlledGate(Z(1, 1, a1 - a0))\\
...         >> H() >> Measure(1)
...     bob = Diagram.swap(qubit, bit) >> bit @ Z(1, 1, b0)\\
...         >> BitControlledGate(Z(1, 1, b1 - b0)) >> H() >> Measure(1)
...     return Model.from_diagram(
...         bit @ Scalar(.5 ** .5) @ Z(0, 2) @ bit >> alice @ bob)
>>> round(bell_test([.3, .05, .17, .42]).contextual_fraction(), 3)
0.17
>>> params, best = ascend(bell_test, [.3, .05, .17, .42])
>>> round(best.contextual_fraction(), 3)
0.414
>>> assert abs(abs(params[1] - params[0]) - .25) < 1e-6

A combinatorial map of recurrent channels has a
:class:`MessagePassingScenario`: its nodes are the parties, each measuring
the system it predicts at the last tick with a local measurement chosen by
a bit, and the messages passed along the edges at the earlier ticks are
what correlates them. Here a source node sends half of a fresh Bell pair
down its edge at every tick and predicts the half it kept a tick later,
when the sink predicts the half it received, so the two predictions are a
Bell pair from the second tick on, and :meth:`MessagePassingScenario.learn`
finds measurements approaching the Tsirelson bound from the most contextual
of a few random starts:

>>> from discopy.optyx import Discard
>>> from discopy.optyx.channel import Ty
>>> from discopy.optyx.qubits import Ket
>>> from discopy.optyx.interaction import Box, CMap
>>> source = Box("source", Ty(), qubit,
...     Discard(qubit) @ qubit >> Scalar(.5 ** .5) @ Z(0, 2) @ qubit,
...     memory=qubit, prediction=qubit)
>>> sink = Box("sink", qubit, Ty(), Ket(0) @ qubit, prediction=qubit)
>>> cmap = CMap([source, sink], [((0, 0), (1, 0))])
>>> scenario = MessagePassingScenario(cmap, Ket(0) ** 3, n_steps=1)
>>> tsirelson = scenario.model([0, 0, .25, .25, .25, .125, 0, .25])
>>> round(tsirelson.contextual_fraction(), 6)
0.414214
>>> angles, learned = scenario.learn(restarts=8, steps=5)
>>> assert learned.contextual_fraction() > .3

The scenario does not depend on the physical system. The same map on
dual-rail photons, where a node predicts a pair of modes and measures it
with a phase, a beam splitter and two photon counters, has pairs of photon
numbers as outcomes and the same contextual fraction, at the angles putting
both settings on a balanced beam splitter a quarter turn of phase apart:

>>> from discopy.optyx import qmode
>>> from discopy.optyx.photonic import Create
>>> pair = qmode ** 2
>>> bell = (Scalar(.5 ** .5) @ Z(0, 2)).to_dual_rail()
>>> source = Box("source", Ty(), pair,
...     Discard(pair) @ pair >> bell @ pair, memory=pair, prediction=pair)
>>> sink = Box("sink", pair, Ty(), Create(0, 0) @ pair, prediction=pair)
>>> photonic = MessagePassingScenario(
...     CMap([source, sink], [((0, 0), (1, 0)), ((0, 1), (1, 1))]),
...     Create(0, 0, 0, 0, 0, 0), n_steps=1)
>>> counts = photonic.model([0, 0, .25, 0, .125, 0, .25, 0])
>>> counts.scenario.outcomes
4
>>> round(counts.contextual_fraction(), 6)
0.414214
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import numpy as np

from discopy.optyx.channel import Diagram, Discard, Ty, bit, qmode, qubit
from discopy.optyx.classical import Bit, BitControlledGate, Digit
from discopy.optyx.core.backends import StateType
from discopy.optyx.photonic import BBS, NumberResolvingMeasurement, Phase
from discopy.optyx.qubits import Measure, X, Z
from discopy.quantum.contextuality import Model, ascend


def distributions(diagram: Diagram, settings: Sequence[int], backend=None,
                  outputs: Sequence[int] = None
                  ) -> list[dict[tuple[int, ...], float]]:
    """
    The probability of each tuple of outcomes of a diagram with one classical
    input and some classical outputs for each party, for each choice of one
    input value below the number of settings of each party, in
    lexicographic order. The probabilities are not normalised, so that a
    postselected diagram is not conditioned on its success, and those below
    :math:`10^{-12}` are left out as numerical noise.

    The diagram is contracted once with its inputs open when the array has
    one entry per choice, which is the case for bits, and once per choice
    otherwise, e.g. for modes, which the open contraction truncates.

    Parameters:
        diagram : A diagram with classical domain and codomain.
        settings : The number of settings of each party, at most two for
            a ``bit`` input.
        backend : The backend evaluating the diagram.
        outputs : The number of output wires of each party, one for each
            by default.

    Example
    -------
    >>> from discopy.optyx import Encode, mode
    >>> from discopy.optyx.photonic import NumberResolvingMeasurement
    >>> counter = Encode(mode) >> NumberResolvingMeasurement(1)
    >>> distributions(counter, settings=(3, ))
    [{(0,): 1.0}, {(1,): 1.0}, {(2,): 1.0}]
    """
    dom, cod = diagram.dom.inside, diagram.cod.inside
    outputs = [1] * len(settings) if outputs is None else list(outputs)
    if not settings:
        raise ValueError("There must be at least one party.")
    if len(dom) != len(settings) or len(cod) != sum(outputs) or not all(
            ob.is_classical for ob in dom + cod):
        raise ValueError(
            "The domain and codomain must be classical types of the length "
            f"of {settings} and {outputs}, got {diagram.dom} and "
            f"{diagram.cod}.")
    if any(n < 1 or ob.name == "bit" and n > 2
           for ob, n in zip(dom, settings)):
        raise ValueError(f"Each party has at least one setting, at most two "
                         f"for a bit input: {settings}.")
    choices = list(product(*map(range, settings)))
    result = diagram.eval(backend)
    array = np.array(result.tensor.array)
    if array.shape[:len(settings)] == tuple(settings):
        if result.state_type is StateType.AMP:
            array = abs(array) ** 2
        tables = [dict(np.ndenumerate(array[choice])) for choice in choices]
    else:
        tables = [(Diagram.tensor(*(
            Bit(x) if ob.name == "bit" else Digit(x)
            for x, ob in zip(choice, dom))) >> diagram
        ).eval(backend).prob_dist(normalise=False) for choice in choices]
    return [{
        tuple(map(int, section)): float(np.real(probability))
        for section, probability in table.items()
        if abs(probability) > 1e-12} for table in tables]


def local_measurement(angles: Sequence[float]) -> Diagram:
    """
    A qubit measurement in one of two bases chosen by a bit, from
    :code:`bit @ qubit` to :code:`bit`, parametrised by four angles in
    turns: the first basis is the computational one rotated by
    :code:`Z(a0)` then :code:`X(b0)`, the second rotated further by
    :code:`Z(a1)` then :code:`X(b1)`. Each basis ranges over the Bloch
    sphere, the first with :code:`b0` as polar and :code:`a0` as
    azimuthal angle.

    >>> assert local_measurement([0, 0, 0, .25]).dom == bit @ qubit
    """
    a0, b0, a1, b1 = angles
    return bit @ (Z(1, 1, a0) >> X(1, 1, b0))\
        >> BitControlledGate(Z(1, 1, a1) >> X(1, 1, b1)) >> Measure(1)


def dual_rail_measurement(angles: Sequence[float]) -> Diagram:
    """
    The photonic counterpart of :func:`local_measurement` on a dual-rail
    qubit, from :code:`bit @ qmode @ qmode` to :code:`mode @ mode`: a
    phase on the first mode then a biased beam splitter, the second
    setting adding a further phase and beam splitter, before counting the
    photons in both modes, so that the outcomes are pairs of photon
    numbers rather than bits.

    >>> from discopy.optyx import mode
    >>> assert dual_rail_measurement([0, 0, 0, .25]).cod == mode @ mode
    """
    a0, b0, a1, b1 = angles
    return bit @ (Phase(a0) @ qmode >> BBS(b0))\
        >> BitControlledGate(Phase(a1) @ qmode >> BBS(b1))\
        >> NumberResolvingMeasurement(2)


class MessagePassingScenario:  # pylint: disable=too-many-instance-attributes
    """
    The measurement scenario of a combinatorial map of recurrent channels,
    :class:`discopy.optyx.interaction.CMap`, whatever the physical systems
    on its wires.

    The parties are the nodes with a prediction, the system a node writes
    to the environment at every tick. Each node has a family of local
    measurements of its prediction, chosen by one classical wire, and a
    context is a choice of one measurement for every node: the predictions
    of a tick are written at once, so any choice can be measured together.
    The empirical model of a choice of measurements is the distribution of
    their outcomes at the last of :code:`n_steps + 1` ticks, every other
    output being discarded. The nodes only interact through the messages
    passed along the edges at the earlier ticks, so the model is contextual
    only through them.

    Parameters:
        cmap : The map.
        initial_state : A state of :code:`cmap.memory` before the first
            tick.
        n_steps : The number of unrollings.
        input_state : A state of :code:`cmap.dom` prepared at every tick,
            required when the map has a boundary.
        measurements : For each node with a prediction, a function from an
            array of angles to a channel from :code:`S @ prediction` to a
            classical type, :code:`S` the single classical wire choosing
            the measurement, with as many output wires for every node. By
            default, :func:`local_measurement` for a qubit and
            :func:`dual_rail_measurement` for a pair of modes.
        sizes : The number of angles of each measurement, required with
            ``measurements`` and four for the defaults.
        settings : The number of measurements of each node, two for each
            by default, which is what a bit chooses between.
        normalise : Whether to condition the model on the success of a
            postselection in the map, see
            :meth:`discopy.quantum.contextuality.Model.normalise`.

    >>> from discopy.optyx.qubits import Ket
    >>> from discopy.optyx.interaction import Box, CMap
    >>> cell = Box("cell", qubit, qubit, Z(2, 3), prediction=qubit)
    >>> ring = CMap([cell, cell], [((0, 1), (1, 0)), ((1, 1), (0, 0))])
    >>> scenario = MessagePassingScenario(ring, Ket(0) ** 4)
    >>> assert scenario.diagram(np.zeros(8)).dom == bit ** 2
    >>> scenario.model(np.zeros(8)).scenario.contexts[0]
    ('a1', 'b1')
    """
    # pylint: disable=too-many-arguments
    # The map, its two states and the number of ticks, then the keywords.
    def __init__(self, cmap, initial_state: Diagram, n_steps: int = 0,
                 input_state: Diagram = None, *,
                 measurements: Sequence[Callable] = None,
                 sizes: Sequence[int] = None, settings: Sequence[int] = None,
                 normalise: bool = False) -> None:
        self.cmap, self.n_steps, self.normalise = cmap, n_steps, normalise
        self.nodes = [box for box in cmap.boxes if box.prediction]
        if measurements is None:
            measurements = [self.default_measurement(box.prediction)
                            for box in self.nodes]
            sizes = [4] * len(self.nodes) if sizes is None else sizes
        if sizes is None or len(sizes) != len(self.nodes)\
                or len(measurements) != len(self.nodes):
            raise ValueError(f"The {len(self.nodes)} nodes with a prediction "
                             "need as many measurements and sizes.")
        if input_state is None:
            if cmap.dom:
                raise ValueError(
                    "input_state is required for a map with a boundary.")
            input_state = Diagram.id()
        self.measurements, self.sizes = list(measurements), list(sizes)
        self.settings = [2] * len(self.nodes)\
            if settings is None else list(settings)
        tick = cmap.cod @ cmap.prediction
        self.state = Diagram.id().tensor(*[input_state] * (n_steps + 1))\
            @ initial_state >> cmap.unroll(n_steps)\
            >> Discard(tick ** n_steps @ cmap.cod)\
            @ Diagram.id(cmap.prediction)

    @staticmethod
    def default_measurement(system: Ty) -> Callable:
        """ The default family of measurements of a system. """
        if system == qubit:
            return local_measurement
        if system == qmode ** 2:
            return dual_rail_measurement
        raise ValueError(f"There is no default measurement of {system}.")

    def measured(self, angles: Sequence[float]) -> list[Diagram]:
        """ The measurement of each node, for a flat array of angles. """
        angles = np.split(np.asarray(angles, dtype=float),
                          np.cumsum(self.sizes)[:-1])
        return [measurement(angle)
                for measurement, angle in zip(self.measurements, angles)]

    def diagram(self, angles: Sequence[float]) -> Diagram:
        """
        The diagram from the settings of every node to their outcomes, for
        a flat array of angles.
        """
        measured = self.measured(angles)
        settings = Ty().tensor(*[m.dom[:1] for m in measured])
        offsets = np.cumsum([len(settings)] + [
            len(box.prediction) for box in self.nodes])
        permutation = [wire for i in range(len(self.nodes)) for wire
                       in [i] + list(range(offsets[i], offsets[i + 1]))]
        return settings @ self.state >> Diagram.permutation(
            permutation, settings @ self.cmap.prediction)\
            >> Diagram.id().tensor(*measured)

    def model(self, angles: Sequence[float], backend=None) -> Model:
        """ The empirical model of a choice of angles. """
        model = Model.from_diagram(
            self.diagram(angles), settings=self.settings, backend=backend,
            outputs=[len(m.cod) for m in self.measured(angles)])
        return model.normalise() if self.normalise else model

    def learn(self, restarts: int = 16, seed: int = 0, backend=None,
              **params) -> tuple[np.ndarray, Model]:
        """
        The angles maximising the contextual fraction: the most contextual
        of :code:`restarts` uniformly random sets of angles is the starting
        point of :func:`discopy.quantum.contextuality.ascend`, which takes
        the remaining keyword arguments. The fraction is zero on the
        non-contextual polytope and the ascent does not move there, which
        is what the restarts are for; when none is contextual, the first
        is returned as it is.
        """
        starts = np.random.default_rng(seed).uniform(
            0, 1, (restarts, sum(self.sizes)))
        fractions = [self.model(start, backend).contextual_fraction()
                     for start in starts]
        return ascend(lambda angles: self.model(angles, backend),
                      starts[int(np.argmax(fractions))], **params)
