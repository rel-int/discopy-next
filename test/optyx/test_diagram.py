import numpy as np
import pytest
from sympy.abc import x, y

from discopy.optyx.core import zw
from discopy.optyx.core.diagram import (
    Bit, Box, Diagram, DualRail, EmbeddingTensor, Mode, Permutation,
    PhotonNumberPreservation, PhotonThresholdDetector, Scalar, Spider, Swap,
    Ty, bit, dual_rail, embedding_tensor, mode)


def test_equal_types_hash_equally():
    assert hash(Bit(2)) == hash(Ty("bit", "bit"))
    assert {Mode(1): "mode"}[mode] == "mode"


def test_a_box_without_array_has_no_semantics():
    box = Box("f", mode, mode)
    for method in (box.conjugate, box.dagger, box.to_path):
        with pytest.raises(NotImplementedError):
            method()
    with pytest.raises(NotImplementedError, match="grad"):
        box.grad(x)
    with pytest.raises(ValueError, match="Input dimensions"):
        box.truncation()
    with pytest.raises(NotImplementedError, match="output_dimensions"):
        box.truncation([2])
    box.photon_preservation_behaviour = PhotonNumberPreservation.CUSTOM
    with pytest.raises(NotImplementedError, match="photon_number_transform"):
        box.photon_number_transform([2], [2])


def test_array_box_semantics():
    """An array is the truncation of a box on bits, and the output
    dimensions of a box on modes when they can be read off its shape."""
    not_ = Box("not", bit, bit, array=np.array([[0, 1], [1, 0]]))
    assert np.allclose(not_.truncation().array, [[0, 1], [1, 0]])
    split = Box("split", mode, mode @ mode, array=np.ones((2, 4)))
    with pytest.raises(NotImplementedError, match="cannot infer"):
        split.determine_output_dimensions([2])


def test_spider_needs_input_dimensions():
    with pytest.raises(ValueError, match="Input dimensions"):
        Spider(1, 2, mode).truncation()


def test_sum_is_linear():
    terms = zw.Endo(2) + zw.Endo(3j)
    assert terms.to_path() == zw.Endo(2 + 3j).to_path()
    assert np.allclose(terms.eval(2).array, 2 ** 2 + 3j ** 2)
    assert terms.conjugate().to_path() == zw.Endo(2 - 3j).to_path()


def test_sum_gradient_term_by_term():
    phases = zw.Endo(np.e ** x) + zw.Endo(np.e ** (2 * x))
    assert phases.grad(y) == Diagram.sum_factory((), mode, mode)
    gradient = phases.grad(x).subs((x, 0))
    assert np.allclose(gradient.eval(1).array, 1 + 2)


def test_permutations_to_path():
    assert np.allclose(
        Permutation(mode ** 3, [1, 2, 0]).to_path().array,
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(Swap(mode, mode).to_path().array, [[0, 1], [1, 0]])


def test_scalar_lambdify():
    assert Scalar(x).lambdify(x)(3) == Scalar(3)


def test_inflation_commutes_with_dagger():
    rail = DualRail(internal_state=[1, 0])
    assert rail.dagger().inflate(2) == rail.inflate(2).dagger()
    detector = PhotonThresholdDetector()
    assert detector.dagger().inflate(2) == detector.inflate(2).dagger()


def test_dual_rail_with_internal_states():
    assert dual_rail(2, internal_states=[[1, 0], [0, 1]]) \
        == DualRail(internal_state=[1, 0]) @ DualRail(internal_state=[0, 1])


def test_embedding_tensors():
    embedding = EmbeddingTensor(2, 3)
    assert np.allclose(embedding.conjugate().array, embedding.array)
    assert np.allclose(
        embedding_tensor(2, 3).eval().array,
        np.einsum("ac,bd->abcd", np.eye(2, 3), np.eye(2, 3)))


def test_feedback_has_no_tensor_until_unrolled():
    loop, = Diagram.swap(mode, mode).feedback().boxes
    with pytest.raises(ValueError, match="no truncation tensor"):
        loop.truncation()
    with pytest.raises(ValueError, match="transforms no photon numbers"):
        loop.photon_number_transform()
