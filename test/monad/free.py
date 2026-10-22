# -*- coding: utf-8 -*-

from pytest import raises

from discopy.kleisli.channel import Channel
from discopy.kleisli.monad import Maybe, Powerset, Subdistribution
from discopy.monad.free import Homomorphism


def test_Homomorphism_id_and_then():
    Free = Homomorphism[Maybe]
    double = Free(lambda x: None if x is None else 2 * x, int, int)
    assert (Free.id(int) >> double)(3) == double(3)\
        == (double >> Free.id(int))(3)
    assert (double >> double)(None) is None
    with raises(TypeError):
        double >> Channel[Maybe](lambda x: x, int, int)


def test_Homomorphism_is_isomorphic_to_Kleisli():
    """
    Extending a channel and restricting it back gives the channel, and the
    extension of a composite of channels is the composite of extensions.
    """
    Free = Homomorphism[Subdistribution]
    flip = Channel[Subdistribution](
        lambda x: frozenset({(x, .5), (not x, .5)}), bool, bool)
    lose = Channel[Subdistribution](
        lambda x: frozenset({(x, .5)}) if x else frozenset({(x, 1.)}),
        bool, bool)
    assert Free.from_channel(flip).to_channel()(True) == flip(True)
    for m in [frozenset({(True, 1.)}), frozenset({(True, .5), (False, .25)})]:
        assert Free.from_channel(flip >> lose)(m)\
            == (Free.from_channel(flip) >> Free.from_channel(lose))(m)
    assert Free.from_channel(Channel[Subdistribution].id(bool))(
        frozenset({(True, .5)})) == Free.id(bool)(frozenset({(True, .5)}))


def test_Homomorphism_from_channel_of_another_monad():
    with raises(TypeError):
        Homomorphism[Powerset].from_channel(
            Channel[Maybe](lambda x: x, int, int))


def test_Homomorphism_repr():
    step = Homomorphism[Powerset](lambda xs: xs, int, int)
    assert repr(step).startswith("monad.free.Homomorphism[Powerset](")
    assert repr(step).endswith("dom=<class 'int'>, cod=<class 'int'>)")
