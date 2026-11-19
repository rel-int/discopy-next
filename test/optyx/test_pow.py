"""
The tensor power of an optyx diagram, i.e. `discopy.monoidal.Ty.__pow__`:
iterated tensor for `n > 0`, the identity for `n <= 0` on an endomorphism
and an assertion on anything else.
"""

import pytest

from discopy.optyx import channel, photonic
from discopy.optyx.core import diagram, zw

ENDOMORPHISMS = [
    zw.Endo(0.5),                                       # core Box
    zw.Merge(2) >> zw.W(2),                             # core Diagram
    photonic.BS >> photonic.BS,                         # channel Diagram
    channel.CQMap(                                      # channel CQMap
        "g", diagram.Id(channel.mode.double()),
        channel.mode, channel.mode),
]

NON_ENDOMORPHISMS = [
    zw.W(2),
    zw.W(2) >> zw.W(2) @ diagram.Id(diagram.mode),
    channel.Measure(channel.qubit),
    channel.CQMap(
        "w", zw.W(2), channel.mode, channel.mode @ channel.mode),
]


@pytest.mark.parametrize("x", ENDOMORPHISMS)
@pytest.mark.parametrize("n", [0, -1, -3])
def test_pow_degenerate_is_id(x, n):
    assert x ** n == x.id(x.dom)


@pytest.mark.parametrize("x", NON_ENDOMORPHISMS)
@pytest.mark.parametrize("n", [0, -1, -3])
def test_pow_degenerate_needs_endomorphism(x, n):
    with pytest.raises(AssertionError):
        x ** n


@pytest.mark.parametrize("x", ENDOMORPHISMS + NON_ENDOMORPHISMS)
def test_pow_positive(x):
    assert x ** 1 == x
    assert x ** 2 == x @ x
    assert x ** 3 == x @ x @ x == x @ (x @ x)


@pytest.mark.parametrize("x", [zw.W(2), photonic.BS])
def test_pow_needs_an_integer(x):
    with pytest.raises(TypeError):
        x ** 1.5


def test_314_degenerate_pow():
    """The two reproducers of rel-int/discopy-next#314's box."""
    with pytest.raises(AssertionError):
        zw.W(2) ** 0
    assert photonic.BS ** 0 == channel.Diagram.id(photonic.BS.dom)
