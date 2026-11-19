import math

import pytest

from discopy.optyx.compiler.mbqc import Fusion, FusionNetwork, Measurement
from discopy.optyx.compiler.patterns import FusionOp, MeasureOp, NextNodeOp
from discopy.optyx.compiler.semm import (
    fn_to_semm,
    get_creation_times,
    get_measurement_times
)


def numeric_order(n: int) -> list[int]:
    return list(range(n + 1))


# Returns a list of unique measurements, all with different angles.
def create_unique_measurements(n: int) -> dict[int, Measurement]:
    small_angle = 2 * math.pi / float(n)
    return {i: Measurement(i * small_angle, "XY") for i in range(n)}


def test_linear_graph_compilation():
    m = create_unique_measurements(3)
    fn = FusionNetwork([0, 1, 2], m, [])

    ins = fn_to_semm(fn, numeric_order)
    assert ins == [
        NextNodeOp(0),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),
        NextNodeOp(2),
        MeasureOp(0, m[2]),
    ]


def test_triangle_compilation():
    m = create_unique_measurements(3)
    fn = FusionNetwork([0, 1, 2], m, [Fusion(0, 2, "X")])

    ins = fn_to_semm(fn, numeric_order)
    assert ins == [
        NextNodeOp(0),
        FusionOp(3, "X"),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),
        NextNodeOp(2),
        FusionOp(0, "X"),
        MeasureOp(0, m[2]),
    ]


@pytest.mark.parametrize("num_measurements", range(1, 3))
def test_linear_graph_measurements(num_measurements: int):
    measurements = create_unique_measurements(num_measurements)
    fn = FusionNetwork([0, 1, 2], measurements, [])

    # Should maybe use the other function
    c = {0: 1, 1: 2, 2: 3}
    m = get_measurement_times(fn, numeric_order, c)
    assert m == {0: 1, 1: 2, 2: 3}


def test_triangle_measurements():
    measurements = create_unique_measurements(3)

    fn = FusionNetwork([0, 1, 2], measurements, [Fusion(0, 2, "X")])

    c = {0: 2, 1: 3, 2: 5}
    m = get_measurement_times(fn, numeric_order, c)

    assert m == {0: 2, 1: 3, 2: 5}


def test_triangle_with_reverse_order_measurements():
    measurements = create_unique_measurements(3)

    fn = FusionNetwork([0, 1, 2], measurements, [Fusion(0, 2, "X")])

    def reverse_order(n: int) -> list[int]:
        return list(range(n, 3))

    c = {0: 2, 1: 3, 2: 5}
    m = get_measurement_times(fn, reverse_order, c)

    assert m == {0: 7, 1: 6, 2: 5}


def test_triangle_with_interesting_order_measurements():
    measurements = create_unique_measurements(3)

    fn = FusionNetwork([0, 1, 2], measurements, [Fusion(0, 2, "X")])

    def custom_order(n: int) -> list[int]:
        if n == 0:
            return [2, 0]
        return [n]

    c = {0: 2, 1: 3, 2: 5}
    m = get_measurement_times(fn, custom_order, c)

    assert m == {0: 6, 1: 3, 2: 5}


def test_creation_times():
    measurements = create_unique_measurements(3)
    fusions = [Fusion(0, 2, "X")]

    fn = FusionNetwork([0, 1, 2], measurements, fusions)

    c = get_creation_times(fn)

    assert c == {0: 2, 1: 3, 2: 5}


def test_creation_times_many_fusions():
    measurements = create_unique_measurements(4)
    fusions = [Fusion(0, 2, "X"), Fusion(0, 3, "X"), Fusion(1, 3, "X")]

    fn = FusionNetwork([0, 1, 2, 3], measurements, fusions)

    c = get_creation_times(fn)

    assert c == {0: 3, 1: 5, 2: 7, 3: 10}


def test_creation_times_skip_the_outputs():
    # An output is not measured, so the machine emits no photon to measure
    # it: the creation times used to count one anyway, which left the fusion
    # photons of every later node off by as many outputs as came before.
    measurements = create_unique_measurements(2)
    fn = FusionNetwork([0, 1, 2, 3], measurements, [Fusion(1, 3, "X")])

    assert get_creation_times(fn) == {0: 1, 1: 3, 2: 3, 3: 4}


def test_fusion_photons_of_a_path_that_is_not_its_ids():
    # The path of a joined cover is not the list of its ids in order, so the
    # times are keyed by node id rather than by position in the path.
    measurements = create_unique_measurements(2)
    fn = FusionNetwork([0, 1, 2, 4, 3], {
        **measurements, 4: Measurement(0, "XY")}, [Fusion(1, 3, "X")])

    assert get_creation_times(fn) == {0: 1, 1: 3, 2: 3, 4: 4, 3: 5}
    ins = fn_to_semm(fn, lambda n: [n])
    assert ins == [
        NextNodeOp(0), MeasureOp(0, measurements[0]),
        NextNodeOp(1), FusionOp(3, "X"), MeasureOp(0, measurements[1]),
        NextNodeOp(2),
        NextNodeOp(4), MeasureOp(0, Measurement(0, "XY")),
        NextNodeOp(3), FusionOp(0, "X")]


def test_fusion_photons_of_parallel_fusions():
    # Two fusions between the same pair of nodes are two photons on each, so
    # the photons are paired by occurrence and not by the pair itself.
    measurements = create_unique_measurements(2)
    fn = FusionNetwork([0, 1], measurements,
                       [Fusion(0, 1, "X"), Fusion(0, 1, "X")])

    assert get_creation_times(fn) == {0: 3, 1: 6}
    assert fn_to_semm(fn, lambda n: [n]) == [
        NextNodeOp(0), FusionOp(3, "X"), FusionOp(3, "X"),
        MeasureOp(0, measurements[0]),
        NextNodeOp(1), FusionOp(0, "X"), FusionOp(0, "X"),
        MeasureOp(0, measurements[1])]
