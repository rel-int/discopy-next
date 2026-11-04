# -*- coding: utf-8 -*-

from math import pi, sqrt

import numpy as np
from pytest import approx, importorskip, mark, raises

from discopy.quantum import CX, H, Ket, Measure, Rz, qubit
from discopy.quantum.contextuality import (
    Inequality, Model, Scenario, ascend, attempt, maximise, sensitivity)

importorskip("scipy")


def measure(phi):
    """ Measurement in the X-Y plane of the Bloch sphere at angle ``phi``. """
    return Rz(-phi / (2 * pi)) >> H >> Measure()


def GHZ(n):
    circuit = Ket(*n * [0]) >> H @ qubit ** (n - 1)
    for i in range(n - 1):
        circuit = circuit >> qubit ** i @ CX @ qubit ** (n - i - 2)
    return circuit


def equatorial(n, phi1, phi2):
    """ The models of Figure 1, with the Bell state for ``n = 2``. """
    return Model.from_circuits(GHZ(n), n * [[measure(phi1), measure(phi2)]])


bell = Scenario.bell(2)
PR = Model(bell, np.array([[1, 0, 0, 1]] * 3 + [[0, 1, 1, 0]]) / 2)
CHSH = Model(bell, np.array([[4, 0, 0, 4]] + 2 * [[3, 1, 1, 3]] + [
    [1, 3, 3, 1]]) / 8)
noise = Model(bell, [[1 / 4] * 4] * 4)


def test_Scenario():
    scenario = Scenario([('x', 'y'), ('y', 'z')], outcomes=3)
    assert scenario.measurements == ('x', 'y', 'z')
    assert len(scenario.assignments) == 3 ** 3
    assert scenario.incidence().shape == (2 * 3 ** 2, 3 ** 3)
    assert (scenario.incidence().sum(axis=0) == 2).all()
    assert eval(repr(scenario)) == scenario
    assert bell.measurements == ('a1', 'b1', 'b2', 'a2')
    assert Scenario.bell(2, settings=(3, 1)).measurements == (
        'a1', 'b1', 'a2', 'a3')
    assert Scenario.bell(3, settings=3).contexts[:2] == (
        ('a1', 'b1', 'c1'), ('a1', 'b1', 'c2'))


def test_Scenario_errors():
    with raises(ValueError):
        bell.relabel({'a1': 'a2'})
    with raises(ValueError):
        bell & bell
    with raises(ValueError):
        bell @ Scenario([('x', )], outcomes=3)
    with raises(ValueError):
        Model(bell, [[1, 0, 0, 0]] * 3)
    with raises(ValueError):
        PR.mix(Model(Scenario([('x', 'y')]), [[1, 0, 0, 0]]), 1 / 2)
    with raises(ValueError):
        Scenario.bell(2, settings=(2, 2, 2))
    with raises(ValueError):
        Inequality(bell, [[1, 0, 0, 0]] * 4)(Model(
            Scenario([('x', 'y')] * 4), [[1, 0, 0, 0]] * 4))
    with raises(ValueError):
        PR.mix(noise, 3 / 2)
    with raises(ValueError):
        Model(bell, [[1.1, -0.1, 0, 0]] + [[1 / 4] * 4] * 3
              ).contextual_fraction()
    for measurements in ([[measure(0)], [measure(0), measure(1)]],
                         [[measure(0)], [measure(0) @ measure(0)]]):
        with raises(ValueError):
            Model.from_circuits(GHZ(2), measurements)


def test_Model_is_compatible():
    model = Model(Scenario([('x', 'y')]), [[.1, .2, .3, .4]])
    assert np.allclose(model.marginal(0, ('y', 'x')), [.1, .3, .2, .4])
    assert np.allclose(model.marginal(0, ('x', )), [.3, .7])
    assert np.allclose(model.marginal(0, ()), [1])
    for model in [PR, CHSH, noise, equatorial(3, 1, 2)]:
        assert model.is_compatible()
    for table in [[[1, 0, 0, 0]] + [[1 / 2, 0, 0, 0]] * 3,
                  [[1.1, -0.1, 0, 0]] + [[1 / 4] * 4] * 3,
                  [[1, 0, 0, 0], [0, 0, 1, 0]] + [[1 / 4] * 4] * 2]:
        model = Model(bell, table)
        assert not model.is_compatible()
        with raises(ValueError):
            model.bell_inequality()
    with raises(ValueError):
        maximise(np.ones(1), np.ones((1, 1)), -np.ones(1), (0, None))


def test_Table_repr():
    assert eval(repr(PR)) == PR and hash(eval(repr(PR))) == hash(PR)
    inequality = CHSH.bell_inequality()
    assert eval(repr(inequality)) == inequality
    assert hash(inequality) == hash(eval(repr(inequality)))
    assert inequality != Inequality(bell, inequality.table, bound=1)


def test_table_I():
    assert PR.contextual_fraction() == approx(1)
    assert Model.from_circuits(GHZ(2), 2 * [
        [measure(0), measure(pi / 3)]]).is_close(CHSH)
    assert CHSH.contextual_fraction() == approx(1 / 4)


def test_table_II():
    e1_SC, e1_NC, e2_SC, e2_NC = (Model(Scenario.bell(3), np.array(
        table) / 16) for table in [
        [[0, 4, 4, 0, 4, 0, 0, 4], [2] * 8, [2] * 8, [4, 0, 0, 4, 0, 4, 4, 0],
         [2] * 8, [4, 0, 0, 4, 0, 4, 4, 0], [4, 0, 0, 4, 0, 4, 4, 0], [2] * 8],
        [[1, 3, 3, 1, 3, 1, 1, 3], [2] * 8, [2] * 8, [3, 1, 1, 3, 1, 3, 3, 1],
         [2] * 8, [3, 1, 1, 3, 1, 3, 3, 1], [3, 1, 1, 3, 1, 3, 3, 1], [2] * 8],
        [[0, 2, 2, 0, 6, 0, 0, 6], [0, 2, 2, 0, 2, 4, 4, 2],
         [0, 2, 2, 0, 2, 4, 4, 2], [2, 0, 0, 2, 0, 6, 6, 0],
         [2, 0, 0, 2, 4, 2, 2, 4], [2, 0, 0, 2, 0, 6, 6, 0],
         [2, 0, 0, 2, 0, 6, 6, 0], [0, 2, 2, 0, 2, 4, 4, 2]],
        [[1, 5, 5, 1, 1, 1, 1, 1], [4, 2, 2, 4, 2, 0, 0, 2],
         [4, 2, 2, 4, 2, 0, 0, 2], [5, 1, 1, 5, 1, 1, 1, 1],
         [2, 4, 4, 2, 0, 2, 2, 0], [5, 1, 1, 5, 1, 1, 1, 1],
         [5, 1, 1, 5, 1, 1, 1, 1], [4, 2, 2, 4, 2, 0, 0, 2]]])
    for model, fraction in zip([e1_SC, e1_NC, e2_SC, e2_NC], [1, 0, 1, 0]):
        assert model.contextual_fraction() == approx(fraction, abs=1e-9)
    mixture = e1_NC.mix(e1_SC, 1 / 2)
    assert mixture == e2_NC.mix(e2_SC, 1 / 2)
    assert mixture.contextual_fraction() == approx(1 / 2)


@mark.parametrize("angles", [(pi / 8, 5 * pi / 8), (7 * pi / 8, 3 * pi / 8)])
def test_table_III(angles):
    p = (sqrt(2) + 2) / 8
    model = equatorial(2, *angles)
    assert model.is_close(Model(bell, [[p, 1 / 2 - p, 1 / 2 - p, p]] + 3 * [
        [1 / 2 - p, p, p, 1 / 2 - p]]))
    assert model.contextual_fraction() == approx(sqrt(2) - 1)
    assert equatorial(2, 0, pi / 2).contextual_fraction() == approx(
        0, abs=1e-9)


@mark.parametrize("n, angles", [
    (3, (pi / 2, 0)), (3, (2 * pi / 3, pi / 6)), (3, (5 * pi / 6, pi / 3)),
    (4, (pi / 2, 0)), (4, (5 * pi / 8, pi / 8)), (4, (3 * pi / 4, pi / 4)),
    (4, (7 * pi / 8, 3 * pi / 8))])
def test_strong_contextuality_of_GHZ(n, angles):
    assert equatorial(n, *angles).contextual_fraction() == approx(1)


@mark.parametrize("n", [3, 4, 5])
def test_proposition_4(n):
    for k in range(n):
        assert equatorial(
            n, (n + k) * pi / (2 * n), k * pi / (2 * n)
        ).contextual_fraction() == approx(1)


@mark.parametrize("model", [
    CHSH, PR, CHSH.mix(noise, 0.9), equatorial(3, 1, 2)])
def test_theorem_1(model):
    fraction = model.contextual_fraction()
    inequality = model.bell_inequality()
    assert inequality.non_contextual_bound == approx(0, abs=1e-9)
    assert inequality.algebraic_bound == approx(1)
    assert inequality.violation(model) == approx(fraction)
    weight, NC, SC = model.decompose()
    assert weight == approx(1 - fraction)
    assert SC.contextual_fraction() == approx(1)
    assert inequality(SC) == approx(1)
    if NC is None:
        assert weight == approx(0, abs=1e-9)
    else:
        assert NC.contextual_fraction() == approx(0, abs=1e-9)
        assert inequality(NC) == approx(0, abs=1e-9)
        assert NC.mix(SC, weight).is_close(model)


def test_theorem_2():
    local = Model(bell, [[1, 0, 0, 0]] * 4)
    weight, NC, SC = local.decompose()
    assert weight == approx(1) and NC.is_close(local) and SC is None
    weight, NC, SC = PR.mix(noise, 1 / 2 + 1e-6).decompose()
    assert NC.mix(SC, weight).is_close(PR.mix(noise, 1 / 2 + 1e-6))
    cd = {'a1': 'c1', 'a2': 'c2', 'b1': 'd1', 'b2': 'd2'}
    e, f, weight = CHSH.mix(local, 1 / 2), PR.relabel(cd), 1 / 3
    CF = Model.contextual_fraction
    assert CF(f) == approx(CF(PR))
    assert CF(CHSH.mix(e, weight)) <= weight * CF(CHSH) + (
        1 - weight) * CF(e) + 1e-9
    assert CF(e & f) == approx(max(CF(e), CF(f)))
    assert CF(e @ f) == approx(CF(e) + CF(f) - CF(e) * CF(f))
    assert CF(CHSH @ CHSH.relabel(cd)) == approx(7 / 16)


triangle = Scenario([('x', 'y'), ('y', 'z'), ('z', 'x')])
specker = Model(triangle, [[0, 1 / 2, 1 / 2, 0]] * 3)


def test_deterministic():
    for scenario in (bell, triangle):
        for assignment in scenario.assignments:
            model = Model.deterministic(scenario, assignment)
            assert model.is_compatible()
            assert model.contextual_fraction() == approx(0, abs=1e-9)
            assert model.bell_inequality().violation(model) == 0
    assert Inequality(bell, [[0] * 4] * 4).violation(PR) == 0
    scaled = CHSH.bell_inequality()
    scaled = Inequality(bell, 1e9 * scaled.vector.reshape(4, 4), bound=0)
    assert scaled.violation(CHSH) == approx(1 / 4)
    assert Inequality(bell, [[1e-12] * 4] * 4).violation(PR) == 0
    with raises(ValueError):
        Inequality(bell, [[1, 0, 0, 0]] * 4, bound=4).violation(
            Model(bell, [[2, 0, 0, 0]] * 4))


@mark.parametrize("weight", np.linspace(0, 1, 11))
def test_isotropic(weight):
    model = PR.mix(noise, weight)
    fraction = model.contextual_fraction()
    assert fraction == approx(max(0, 2 * weight - 1), abs=1e-9)
    assert model.bell_inequality().violation(model) == approx(
        fraction, abs=1e-9)


def test_specker():
    assert specker.is_compatible()
    assert specker.contextual_fraction() == approx(1)
    inequality = specker.bell_inequality()
    assert inequality.non_contextual_bound == approx(0, abs=1e-9)
    assert inequality.violation(specker) == approx(1)


@mark.parametrize("model", [CHSH, PR.mix(noise, .9), equatorial(3, 1, 2)])
def test_gradient(model):
    gradient = model.gradient()
    assert gradient.scenario == model.scenario
    assert gradient.vector @ model.vector == approx(
        model.contextual_fraction() - 1)
    for assignment in model.scenario.assignments:
        deterministic = Model.deterministic(model.scenario, assignment)
        assert gradient.vector @ deterministic.vector <= -1 + 1e-9
    incidence = model.scenario.incidence()
    assert np.allclose(gradient.vector, -sensitivity(
        np.ones(incidence.shape[1]), incidence, model.vector, (0, None)))
    with raises(ValueError):
        Model(bell, [[1, 0, 0, 0], [0, 0, 1, 0]] + [[1 / 4] * 4] * 2
              ).gradient()


def test_ascend():
    isotropic = lambda params: PR.mix(noise, min(1., max(0., params[0])))
    weight, box = ascend(isotropic, [.75], steps=5)
    assert box.is_close(PR) and weight[0] >= 1
    weight, model = ascend(isotropic, [.25], steps=2)
    assert weight[0] == approx(.25) and model.is_close(isotropic([.25]))
    unclamped = lambda params: PR.mix(noise, params[0])
    weight, box = ascend(unclamped, [.75], steps=5)
    assert .99 < weight[0] <= 1 and box.contextual_fraction() > .98
    assert attempt(unclamped, np.array([1.5])) is None
    assert attempt(lambda params: Model(bell, [[1, 0, 0, 0], [0, 0, 1, 0]]
                   + [[1 / 4] * 4] * 2), np.zeros(1)) is None


def test_normalise():
    half = Model(bell, [[1 / 4, 0, 0, 1 / 4]] * 3 + [[0, 1 / 4, 1 / 4, 0]])
    assert half.normalise().is_close(PR)
    with raises(ValueError):
        Model(bell, [[1 / 4, 0, 0, 1 / 4]] * 3 + [[0, 1, 1, 0]]).normalise()
    with raises(ValueError):
        Model(bell, [[0] * 4] * 4).normalise()
    rare = Model(bell, [[1e-12, 0, 0, 1e-12]] * 3 + [[0, 1e-12, 1e-12, 0]])
    assert rare.normalise().is_close(PR)
    with raises(ValueError):
        Model(bell, [[1 / 2, 0, 0, 0], [0, 0, 1 / 2, 0]]
              + [[1 / 8] * 4] * 2).normalise()
