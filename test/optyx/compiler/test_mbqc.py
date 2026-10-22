import pytest

from discopy.optyx.compiler.mbqc import Measurement, pattern_satisfies_order


def test_measurement_plane_must_exist():
    with pytest.raises(ValueError, match="unexpected measurement plane"):
        Measurement(0.5, "XX").to_graphix()


def test_measuring_before_the_past_breaks_the_order():
    meas = Measurement(0, "XY")
    order = lambda n: list(range(n))
    assert pattern_satisfies_order([(0, meas), (1, meas)], order)
    assert not pattern_satisfies_order([(1, meas), (0, meas)], order)
