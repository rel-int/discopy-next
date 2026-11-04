from math import sqrt

from pytest import approx, raises

import numpy as np

from discopy.optyx import Diagram, Discard, Encode, bit, mode, qmode, qubit
from discopy.optyx.channel import Ty
from discopy.optyx.classical import BitControlledGate
from discopy.optyx.contextuality import (
    MessagePassingScenario, distributions, local_measurement)
from discopy.optyx.interaction import Box, CMap
from discopy.optyx.photonic import Create, NumberResolvingMeasurement
from discopy.optyx.qubits import H, Ket, Measure, Scalar, Z
from discopy.quantum.contextuality import Model, Scenario


alice = BitControlledGate(Z(1, 1, .25)) >> H() >> Measure(1)
bob = Diagram.swap(qubit, bit) >> bit @ Z(1, 1, -.125)\
    >> BitControlledGate(Z(1, 1, .25)) >> H() >> Measure(1)
bell_test = bit @ Scalar(.5 ** .5) @ Z(0, 2) @ bit >> alice @ bob
detector = Encode(mode) >> NumberResolvingMeasurement(1)


def test_from_diagram():
    tsirelson = Model.from_diagram(bell_test)
    p = (sqrt(2) + 2) / 8
    assert tsirelson.is_close(Model(Scenario.bell(2), 3 * [
        [p, 1 / 2 - p, 1 / 2 - p, p]] + [[1 / 2 - p, p, p, 1 / 2 - p]]))
    assert tsirelson.contextual_fraction() == approx(sqrt(2) - 1)
    assert Model.from_diagram(bell_test, outcomes=3).contextual_fraction(
        ) == approx(sqrt(2) - 1)
    counter = Model.from_diagram(detector @ detector, settings=(3, 1))
    assert counter.scenario == Scenario.bell(2, (3, 1), outcomes=3)
    assert counter.table[2] == (0, 0, 0, 0, 0, 0, 1, 0, 0)
    assert counter.contextual_fraction() == approx(0, abs=1e-9)


def test_from_diagram_unnormalised():
    unnormalised = bit @ Z(0, 2) @ bit >> alice @ bob
    assert all(sum(row.values()) == approx(2)
               for row in distributions(unnormalised, (2, 2)))
    model = Model.from_diagram(unnormalised)
    assert not model.is_compatible()
    with raises(ValueError):
        model.contextual_fraction()


def test_from_diagram_errors():
    with raises(ValueError):
        Model.from_diagram(Measure(1))
    with raises(ValueError):
        Model.from_diagram(detector @ Encode(mode))
    with raises(ValueError):
        distributions(detector, (2, 2))
    with raises(ValueError):
        Model.from_diagram(bell_test, settings=(3, 2))
    with raises(ValueError):
        Model.from_diagram(bell_test, settings=(0, 2))
    with raises(ValueError):
        Model.from_diagram(detector @ detector, settings=(3, 1), outcomes=2)


def test_distributions_open_and_closed():
    from discopy.optyx.classical import Bit
    rows = distributions(bell_test, (2, 2))
    for choice, row in zip([(a, b) for a in range(2) for b in range(2)], rows):
        closed = (Bit(choice[0]) @ Bit(choice[1]) >> bell_test).eval(
            ).prob_dist()
        assert all(closed[section] == approx(p) for section, p in row.items())
    with raises(ValueError):
        distributions(bell_test, (2, 2), outputs=(2, 2))
    assert distributions(Diagram.id(bit), (2, )) == [{(0, ): 1}, {(1, ): 1}]
    with raises(ValueError):
        Model.from_diagram(bell_test, outputs=(1, 2))
    with raises(ValueError):
        Model.from_diagram(bell_test, outputs=(1, ))
    with raises(ValueError):
        distributions(Diagram.id(), ())


def test_local_measurement():
    plus = Ket(0) >> H()
    measured = distributions(
        bit @ plus >> local_measurement([0, 0, .25, .25]), (2, ))
    assert list(measured[0].values()) == approx([.5, .5])
    assert len(measured[1]) == 1


def bell_pair_map():
    source = Box("source", Ty(), qubit,
                 Discard(qubit) @ qubit >> Scalar(.5 ** .5) @ Z(0, 2) @ qubit,
                 memory=qubit, prediction=qubit)
    sink = Box("sink", qubit, Ty(), Ket(0) @ qubit, prediction=qubit)
    return CMap([source, sink], [((0, 0), (1, 0))])


def test_MessagePassingScenario():
    scenario = MessagePassingScenario(bell_pair_map(), Ket(0) ** 3, n_steps=1)
    angles = [0, 0, .25, .25, .25, .125, 0, .25]
    assert scenario.diagram(angles).dom == scenario.diagram(angles).cod\
        == bit ** 2
    assert scenario.model(angles).contextual_fraction() == approx(sqrt(2) - 1)
    first_tick = MessagePassingScenario(bell_pair_map(), Ket(0) ** 3)
    assert first_tick.model(angles).contextual_fraction() == approx(
        0, abs=1e-9)
    with raises(ValueError):
        MessagePassingScenario(CMap([Box(
            "f", qubit, Ty(), Z(1, 3), prediction=qubit ** 2)], []), Ket(0))
    with raises(ValueError):
        MessagePassingScenario(
            CMap([Box("f", qubit, Ty(), Diagram.id(qubit))], []),
            Diagram.id())
    with raises(ValueError):
        MessagePassingScenario.default_measurement(mode)
    custom = MessagePassingScenario(
        bell_pair_map(), Ket(0) ** 3, n_steps=1,
        measurements=[local_measurement] * 2, sizes=[4, 4])
    assert custom.model(angles).is_close(scenario.model(angles))
    for kwargs in [dict(measurements=[local_measurement] * 2),
                   dict(sizes=[4]), dict(measurements=[local_measurement],
                                         sizes=[4])]:
        with raises(ValueError):
            MessagePassingScenario(bell_pair_map(), Ket(0) ** 3, **kwargs)


def test_learn():
    scenario = MessagePassingScenario(bell_pair_map(), Ket(0) ** 3, n_steps=1)
    angles, model = scenario.learn(restarts=2, steps=1)
    assert angles.shape == (8, )
    assert model.is_close(scenario.model(angles))


def test_dual_rail_scenario():
    pair = qmode ** 2
    bell = (Scalar(.5 ** .5) @ Z(0, 2)).to_dual_rail()
    source = Box("source", Ty(), pair, Discard(pair) @ pair >> bell @ pair,
                 memory=pair, prediction=pair)
    sink = Box("sink", pair, Ty(), Create(0, 0) @ pair, prediction=pair)
    cmap = CMap([source, sink], [((0, 0), (1, 0)), ((0, 1), (1, 1))])
    photonic = MessagePassingScenario(
        cmap, Create(0, 0, 0, 0, 0, 0), n_steps=1)
    counts = photonic.model([0, 0, .25, 0, .125, 0, .25, 0])
    assert counts.scenario == Scenario.bell(2, outcomes=4)
    assert counts.is_compatible()
    assert counts.contextual_fraction() == approx(sqrt(2) - 1)
    conditioned = MessagePassingScenario(
        cmap, Create(0, 0, 0, 0, 0, 0), n_steps=1, normalise=True)
    assert conditioned.model(np.zeros(8)).is_close(photonic.model(np.zeros(8)))
