import numpy as np
import pytest
from sympy import Expr
from sympy.abc import x, y

from discopy import optyx
from discopy.utils import AxiomError
from discopy.optyx.core import control, diagram, zw, zx
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


def test_sum_is_linear_on_one_photon_only():
    terms = zw.Endo(2) + zw.Endo(3j)
    assert terms.to_path() == (
        zw.Endo(2).to_path() + zw.Endo(3j).to_path())
    assert np.allclose(terms.eval(1).array,
                       zw.Endo(2 + 3j).to_path().eval(1).array)
    assert np.allclose(terms.eval(2).array, 2 ** 2 + 3j ** 2)
    assert not np.allclose(terms.eval(2).array, (2 + 3j) ** 2)
    assert terms.conjugate().to_path() == (
        zw.Endo(2) + zw.Endo(-3j)).to_path()


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


def test_symbolic_path_through_a_swap():
    """A path matrix of expressions composes with the swap blocks that
    creations, selections and swaps bring in, and a permutation's path
    matrix has the dtype it is asked for."""
    circuit = zw.Create(1, 0) >> zw.Endo(x) @ zw.Id(1) \
        >> zw.SWAP >> zw.Select(0, 1)
    assert float(circuit.to_path(Expr).eval().array.item().subs(x, 2)) == 2
    for permutation in (Swap(mode, mode), Permutation(mode ** 3, [1, 2, 0])):
        assert permutation.to_path(Expr).dtype is Expr


def test_scalar_lambdify():
    assert Scalar(x).lambdify(x)(3) == Scalar(3)


def test_scalar_of_any_symbolic_expression():
    for expr in (x ** 2, x + 1, x + y):
        assert Scalar(expr).scalar == expr
    assert Scalar(x - x + 2).scalar == 2


def test_inflation_commutes_with_dagger():
    rail = DualRail(internal_state=[1, 0])
    assert rail.dagger().inflate(2) == rail.inflate(2).dagger()
    detector = PhotonThresholdDetector()
    assert detector.dagger().inflate(2) == detector.inflate(2).dagger()


def test_inflation_fixes_scalars():
    assert Scalar(2).inflate(3) == Scalar(2)


def test_inflation_is_linear():
    terms = zw.Endo(2) + zw.Endo(3)
    assert terms.inflate(2) == zw.Endo(2).inflate(2) + zw.Endo(3).inflate(2)


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


DAGGERABLE = [
    Box("f", bit, bit @ bit, array=np.arange(8).reshape(2, 2, 2) * 1j),
    zx.ZXBox("g", 1, 2, array=np.arange(8).reshape(2, 2, 2) * 1j),
    zw.ZWBox("h", mode, mode, array=np.arange(9).reshape(3, 3) * 1j),
    control.ClassicalBox("k", bit, bit, array=[[0, 1], [1, 0]]),
    control.ControlBox("c", bit, bit, array=[[0, 1j], [1, 0]]),
    Box("s", Ty(), bit @ bit, array=np.arange(4).reshape(2, 2) * 1j),
    Box("e", bit @ bit, Ty(), array=np.arange(4).reshape(2, 2) * 1j),
]


def adjoint(box):
    """The conjugate of the array with its codomain axes moved first, or
    its conjugate transpose when it is a matrix."""
    dom, cod, array = len(box.dom), len(box.cod), np.asarray(box.array)
    if array.ndim != dom + cod:
        return array.T.conj()
    return np.conj(np.moveaxis(
        array, range(dom + cod), [*range(cod, dom + cod), *range(cod)]))


@pytest.mark.parametrize("box", DAGGERABLE, ids=lambda box: box.name)
def test_the_dagger_of_the_dagger_is_the_box(box):
    """rel-int/discopy-next#497: the dagger flips ``is_dagger`` as in
    :mod:`discopy.cat`, so its dagger is the box, and its array is the
    adjoint once."""
    dagger = box[::-1]
    assert dagger.name == box.name and dagger.is_dagger
    assert str(dagger) == box.name + "[::-1]"
    assert dagger[::-1] == box and (dagger[::-1].array == box.array).all()
    assert type(dagger) is type(box) and not dagger[::-1].is_dagger
    assert (dagger.array == adjoint(box)).all()
    dims = [3] * len(box.dom) if box.dom == mode else None
    tensor = box.truncation(dims, dims)
    assert dagger.truncation(dims, dims) == tensor[::-1]
    assert np.allclose(
        dagger.truncation(dims, dims).eval().array,
        tensor.eval().dagger().array)
    assert np.allclose(
        dagger.truncation(dims, dims).eval().array.reshape(-1),
        adjoint(box).reshape(-1))


def test_the_dagger_of_a_state_infers_its_dimension():
    state = Box("s", Ty(), mode, array=[1, 2, 3])
    assert state[::-1].truncation() == state.truncation()[::-1]
    assert state[::-1].truncation().eval().array.shape == (3, )


@pytest.mark.parametrize("box", DAGGERABLE, ids=lambda box: box.name)
def test_the_dagger_reads_back(box):
    namespace = {**vars(diagram), **vars(zx), **vars(zw), **vars(control),
                 "optyx": optyx}
    assert eval(repr(box[::-1]), namespace) == box[::-1]
    assert (eval(repr(box[::-1]), namespace).array == box[::-1].array).all()


def test_a_weight_has_no_dagger():
    with pytest.raises(NotImplementedError, match="dagger"):
        control.Weight(lambda n: 1, mode)[::-1]


def test_a_self_adjoint_box_reads_back():
    assert zx.H[::-1] is zx.H
    assert eval(repr(zx.H), {**vars(zx), "optyx": optyx}) == zx.H


def test_the_dagger_is_taken_by_the_backend_of_its_array():
    """The adjoint of the array keeps its type and its gradient, whichever
    backend is active."""
    torch = pytest.importorskip("torch")
    from discopy.matrix import backend
    array = torch.arange(8.).reshape(2, 2, 2).requires_grad_()
    dagger = Box("f", bit, bit @ bit, array=array)[::-1].array
    assert dagger.requires_grad and dagger[1, 0, 0] == array[0, 1, 0]
    with backend("pytorch"):
        dagger = Box("f", bit, bit @ bit,
                     array=np.arange(8).reshape(2, 2, 2))[::-1].array
    assert isinstance(dagger, np.ndarray) and dagger[1, 0, 0] == 2


def test_a_wire_bends_through_the_spider_with_two_legs():
    """The cup and cap of an optyx type are its own spiders, so the snake
    equations hold where they used to raise (next#661)."""
    assert Diagram.cups(mode, mode) == Spider(2, 0, mode)
    assert Diagram.caps(bit, bit) == Spider(0, 2, bit)
    left = Diagram.caps(bit, bit) @ bit >> bit @ Diagram.cups(bit, bit)
    right = bit @ Diagram.caps(bit, bit) >> Diagram.cups(bit, bit) @ bit
    for snake in (left, right):
        assert np.allclose(
            snake.to_tensor().eval().array.reshape(2, 2), np.eye(2))
    with pytest.raises(AxiomError, match="not adjoints"):
        Diagram.cups(mode, bit)


DIAGRAM_BOXES = [
    Box("f", bit, mode),
    Box("f", bit, bit @ bit, array=np.arange(8).reshape(2, 2, 2) * 1j),
    Box("f", bit, bit @ bit,
        array=np.arange(8).reshape(2, 2, 2) * 1j).dagger(),
    Box("h", bit, bit, array=np.eye(2), is_dagger=None),
    DualRail(),
    DualRail(is_dagger=True, internal_state=[1, 0]),
    EmbeddingTensor(2, 4),
    EmbeddingTensor(1, 3),
    Permutation(mode ** 3, [1, 2, 0]),
    PhotonThresholdDetector(),
    PhotonThresholdDetector(is_dagger=True),
    Scalar(0.5 + 1j),
    Scalar(x),
    Spider(1, 2, mode),
    Spider(1, 2, bit, 0.25, is_dagger=True),
    Swap(mode, bit),
    Swap(mode, mode).feedback(state=Spider(0, 1, mode)),
    Scalar(1) + Scalar(2),
]


@pytest.mark.parametrize("box", DIAGRAM_BOXES, ids=repr)
def test_a_diagram_box_loads_its_own_tree(box):
    from discopy.utils import dumps, loads
    loaded = loads(dumps(box))
    assert loaded == box
    assert loaded.to_tree() == box.to_tree()
    assert loaded.is_dagger is box.is_dagger
    if getattr(box, "array", None) is not None:
        assert np.array_equal(loaded.array, box.array)
    if isinstance(box, Diagram):
        assert loads(dumps(box @ box)) == box @ box
