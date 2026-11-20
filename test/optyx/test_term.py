import numpy as np
import pytest

from discopy.optyx import photonic
from discopy.optyx.term import (
    Term, Box, Contraction, Dereliction, Digging, Exchange, Laxator,
    Weakening, qubit)
from discopy.optyx.utils.misc import compare_arrays_of_different_sizes

theta, phi = .3, .7
U = np.array([[np.cos(theta), -np.exp(1j * phi) * np.sin(theta)],
              [np.sin(theta), np.exp(1j * phi) * np.cos(theta)]])
u = Box("U", qubit, qubit, data=U)


def amplitudes(protocol):
    return protocol.get_kraus().to_tensor().eval().array


@pytest.mark.parametrize("term", [
    u.bang() >> Dereliction(qubit), Dereliction(qubit) >> u])
@pytest.mark.parametrize("i", [0, 1])
def test_dereliction(term, i):
    state = photonic.Create(*[(1, 0), (0, 1)][i])
    assert np.allclose(amplitudes(state >> term.to_channel()), U[:, i])


@pytest.mark.parametrize("photons", [(1, 1), (2, 1), (0, 3)])
def test_contraction(photons):
    state = photonic.Create(*photons)
    assert compare_arrays_of_different_sizes(
        amplitudes(state >> (u.bang() >> Contraction(qubit)).to_channel()),
        amplitudes(state >> (
            Contraction(qubit) >> u.bang() @ u.bang()).to_channel()))


def test_from_callable():
    @Term.from_callable(qubit.bang() @ qubit, qubit @ qubit)
    def term(x, y):
        x, z = Contraction(qubit)(u.bang()(x))
        Weakening(qubit)(z)
        return Dereliction(qubit)(x), u(y)

    zero = photonic.Create(1, 0) >> photonic.DualRail(1).dagger()
    state = photonic.Create(1, 0) @ zero
    assert isinstance(term, Term)
    assert np.allclose(amplitudes(state >> term.to_channel()),
                       np.outer(U[:, 0], U[:, 0]))


@pytest.mark.parametrize("term", [
    Digging(qubit), Laxator(), Laxator(qubit, qubit),
    Box("f", qubit @ qubit, qubit), (Dereliction(qubit) >> u).bang(),
    Term.id((qubit @ qubit).bang())])
def test_no_optical_image(term):
    with pytest.raises(NotImplementedError):
        term.to_channel()


flip = Box("X", qubit, qubit, data=np.array([[0, 1], [1, 0]]))


@pytest.mark.parametrize("term", [
    flip, flip.bang(), Contraction(qubit), Weakening(qubit),
    Dereliction(qubit), Exchange(qubit, qubit, True), Digging(qubit),
    Laxator(), Contraction(qubit) >> flip.bang() @ Weakening(qubit)])
def test_transparency(term):
    from numpy import array  # noqa: F401
    from discopy import bang, monoidal, optyx  # noqa: F401
    assert eval(repr(term)) == term


def test_kappa_term():
    H = Box("H", qubit, qubit, data=np.array([[1, 1], [1, -1]]) / 2 ** .5)
    copy_H = qubit.bang()(lambda x: Contraction(qubit)(H.bang()(x)))
    proof = copy_H.eval() >> Dereliction(qubit) @ Dereliction(qubit)
    assert isinstance(proof, Term)
    bell = amplitudes(photonic.Create(1, 1) >> proof.to_channel())
    assert np.allclose(bell, [[1, 0], [0, -1]])
    term = qubit.bang()(lambda x: Dereliction(qubit)(u.bang()(x)))
    assert np.allclose(amplitudes(
        photonic.Create(0, 1) >> term.eval().to_channel()), U[:, 1])


def test_kappa_weakening():
    term = (qubit.bang() @ qubit.bang())(
        lambda a, b: Dereliction(qubit)(u.bang()(b)))
    channel = term.eval().to_channel()
    vacuum, photon = photonic.Create(0, 0), photonic.Create(1, 0)
    assert np.allclose(amplitudes(vacuum @ photon >> channel), U[:, 0])
    assert np.allclose(amplitudes(photon @ photon >> channel), 0)
