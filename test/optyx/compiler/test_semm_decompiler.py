import pytest

from discopy.optyx.compiler.mbqc import Measurement
from discopy.optyx.compiler.patterns import FusionOp, MeasureOp, NextNodeOp
from discopy.optyx.compiler.semm_decompiler import (
    ValidationError, decompile_to_fusion_network)


MEASUREMENT = Measurement(0.5, "XY")


# The streams below are boxes of next#314: each decompiled into a fusion
# network, two of them silently and one with an IndexError.
@pytest.mark.parametrize("instructions", [
    [MeasureOp(0, MEASUREMENT)],
    [MeasureOp(3, MEASUREMENT)],
    [FusionOp(0, "X")],
    [FusionOp(3, "X")],
])
def test_an_instruction_before_the_first_node_is_invalid(instructions):
    with pytest.raises(ValidationError, match="no node to operate on"):
        decompile_to_fusion_network(instructions)


@pytest.mark.parametrize("instructions", [
    [NextNodeOp(0), MeasureOp(3, MEASUREMENT), MeasureOp(0, MEASUREMENT)],
    [NextNodeOp(0), MeasureOp(0, MEASUREMENT), MeasureOp(3, MEASUREMENT)],
])
def test_a_node_measured_twice_through_a_delay_is_invalid(instructions):
    with pytest.raises(ValidationError, match="already measured node 0"):
        decompile_to_fusion_network(instructions)


def test_a_delayed_fusion_nobody_comes_back_for_is_invalid():
    with pytest.raises(
            ValidationError, match="no photon exits the delay loop"):
        decompile_to_fusion_network(
            [NextNodeOp(0), FusionOp(5, "X"), MeasureOp(0, MEASUREMENT)])
