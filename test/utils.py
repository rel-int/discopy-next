from os import listdir

import pickle
import re

import pytest
from pytest import warns

from unittest.mock import MagicMock
from unittest.mock import patch

from discopy import cat, rigid
from discopy.cat import Ob
from discopy.utils import *
from discopy.tensor import Box

import pytest
from pytest import warns

from os import listdir
import pickle

zip_mock = MagicMock()
zip_mock.open().__enter__().read.return_value =\
    '[{"factory": "cat.Ob", "name": "a"}]'


@patch('urllib.request.urlretrieve', return_value=(None, None))
@patch('zipfile.ZipFile', return_value=zip_mock)
def test_load_corpus(a, b):
    assert load_corpus("[fake url]") == [Ob("a")]


def test_deprecated_from_tree():
    tree = {
        'factory': 'discopy.rigid.Diagram',
        'dom': {'factory': 'discopy.rigid.Ty',
                'objects': [{'factory': 'discopy.rigid.Ob', 'name': 'n'}]},
        'cod': {'factory': 'discopy.rigid.Ty',
                'objects': [{'factory': 'discopy.rigid.Ob', 'name': 'n'}]},
        'boxes': [], 'offsets': []}
    with warns(DeprecationWarning):
        assert from_tree(tree) == rigid.Id(rigid.Ty('n'))


def test_named_generic_cache():
    from discopy import tensor as dt
    box, box_int, box_float = dt.Box, dt.Box[int], dt.Box[float]
    assert box_int is dt.Box[int]
    assert box is not box_int and box_float is not box_int
    diag_int = dt.Diagram[int]
    assert diag_int is dt.Diagram[int]
    assert box_int is dt.Box[int]



def _rounded_repr(obj):
    # Gate matrices such as the Hadamard's 1 / sqrt(2) entries are stored as
    # floats whose last bit depends on the numpy version that generated the
    # pickle, so exact equality across versions is not portable. Round every
    # float in the repr to 12 significant figures before comparing.
    return re.sub(
        r'\d+\.\d+', lambda m: format(float(m.group()), '.12g'), repr(obj))


@pytest.mark.parametrize('version', ['0.6', '1.2'])
@pytest.mark.parametrize('fn', listdir('test/fixtures/pickles/1.3/'))
def test_pickle_version_compatibility(fn, version):
    if fn == 'quantum.Circuit.pickle':
        pytest.importorskip("pytket")
    with open(f"test/fixtures/pickles/1.3/{fn}", 'rb') as f:
        new = pickle.load(f)
    with open(f"test/fixtures/pickles/{version}/{fn}", 'rb') as f:
        old = pickle.load(f)
    assert old == new or _rounded_repr(old) == _rounded_repr(new)


def test_parameterised_box_pickle():
    box = Box("A", 2, 3)
    assert pickle.loads(pickle.dumps(box)) == box


def test_named_generic_pickle():
    from copy import deepcopy
    from discopy import interaction, matrix, tensor
    from discopy.cmap import CMap
    from discopy.hypergraph import Hypergraph
    from discopy.monoidal import List
    from discopy.symmetric import Diagram, Ty
    x = Ty('x')
    for term in (interaction.Ty[int](1, 2), Hypergraph[Diagram].id(x),
                 List[int](2, 3), matrix.Matrix[int]([1, 0, 0, 1], 2, 2),
                 tensor.Tensor[complex]([1j], tensor.Dim(1), tensor.Dim(1)),
                 CMap[Diagram].id(x)):
        loaded = pickle.loads(pickle.dumps(term))
        assert "__class_getitem__values__" not in vars(term)
        assert type(loaded) is type(term) and loaded == term
        copied = deepcopy(term)
        assert type(copied) is type(term) and copied == term


def test_named_generic_pickle_subscript():
    """ A class subscripted by a subscripted class pickles too. """
    from discopy import tensor
    from discopy.axioms import Equation
    from discopy.hypergraph import Hypergraph
    from discopy.matrix import Matrix
    m = Matrix[int].id(2)
    for term in (Equation[Matrix[int]](m, m),
                 Hypergraph[tensor.Diagram[complex]].id(tensor.Dim(2))):
        loaded = pickle.loads(pickle.dumps(term))
        assert type(loaded) is type(term)
        assert vars(loaded) == vars(term)


def test_deprecated_ob():
    from discopy import (
        biclosed, braided, compact, feedback, frobenius, pivotal, rigid)
    from discopy.grammar import pregroup
    from discopy.quantum import circuit
    for module in (rigid, braided, biclosed, pivotal, frobenius, feedback,
                   circuit, pregroup, compact):
        with warns(DeprecationWarning):
            assert module.Ob is module.Wire
        with pytest.raises(AttributeError):
            module.not_an_attribute


def test_wire_tree_roundtrip():
    from discopy import biclosed, braided, feedback, frobenius, pivotal, rigid
    from discopy.quantum import circuit
    for x in (rigid.Wire('x'), braided.Wire('x'), biclosed.Wire('x'),
              pivotal.Wire('x'), frobenius.Wire('x'), feedback.Wire('x'),
              circuit.Digit(2)):
        assert from_tree(x.to_tree()) == x
    with warns(DeprecationWarning):
        assert from_tree({'factory': 'discopy.frobenius.Ob', 'name': 'x'})\
            == frobenius.Wire('x')


def test_NamedGeneric_factory():
    """ A subscript of a factory is its own; the others keep ar as itself. """
    from discopy import symmetric
    from discopy.abc import Category
    from discopy.cmap import CMap
    from discopy.hypergraph import Hypergraph
    from discopy.utils import NamedGeneric, factory

    @factory
    class F(Category, NamedGeneric["dtype"]):
        """ The arrows of a category parameterised by a dtype. """

    class G(F):
        """ A box of F, which is not a factory. """

    class H(Category, NamedGeneric["dtype"]):
        """ A category that is not a factory, like Hypergraph. """

    assert F[int].factory is F[int] is F[int].ar
    assert G[int].ar is F
    assert H[int].ar is H[int] and not hasattr(H[int], "factory")
    for wiring in (Hypergraph, CMap):
        assert wiring[symmetric.Diagram].ar is wiring[symmetric.Diagram]


def test_resolve_factory():
    from discopy import tensor, monoidal
    from discopy.utils import resolve_factory, factory_name
    for cls in (tensor.Diagram[complex], monoidal.List[int], rigid.Ty):
        assert resolve_factory(factory_name(cls)) is cls
    assert resolve_factory("complex") is complex
    with pytest.raises(AttributeError):
        resolve_factory("nonsense")


def test_MappingOrCallable():
    assert MappingOrCallable({0: 1})[0] == MappingOrCallable(lambda x: 1)[0]
    with pytest.raises(TypeError):
        MappingOrCallable([10, 20])


def test_value_errors_have_messages():
    from discopy import closed, lens, symmetric
    x, y = closed.Ty('x'), closed.Ty('y')
    for function, message in [
            (lambda: rigid.Box('f', 'x', 'y').curry(2), "split 2"),
            (lambda: closed.Box('f', x, y).uncurry(), "exponential"),
            (lambda: closed.Box('f', x, y << x @ x).uncurry(1), "uncurry 1"),
            (lambda: closed.Box('f', x, y << x).uncurry(-1), "split -1"),
            (lambda: lens.Symmetric[symmetric.Diagram].permutation(
                [0, 0], [x, x]), "permutation"),
            (lambda: pushout(2, 3, [1], [0, 1]), "equal length"),
            (lambda: assert_istraceable(symmetric.Box('f', 'x', 'x'), 2),
             "split 2"),
            (lambda: inductive(lambda n: n)(0, -1), "non-negative")]:
        with pytest.raises(ValueError, match=message):
            function()


def test_NamedGeneric_without_type_parameter():
    class X(NamedGeneric):
        """ A class that declares no type parameter. """

    with pytest.raises(TypeError):
        X[int]

def test_encode_complex():
    assert encode_complex(1 + 2j) == {
        'factory': 'complex', 're': 1.0, 'im': 2.0}
    assert encode_complex(complex(0)) == {
        'factory': 'complex', 're': 0.0, 'im': 0.0}
    with pytest.raises(TypeError) as err:
        encode_complex(Ob('x'))
    assert "Ob" in str(err.value)


def test_decode_complex():
    assert decode_complex(encode_complex(1 + 2j)) == 1 + 2j
    for tree in ({'factory': 'cat.Ob', 'name': 'x'},
                 {'re': 1, 'im': 2},
                 {'factory': 'complex', 're': 1},
                 {'factory': 'complex', 're': 1, 'im': 2, 'name': 'x'}):
        assert decode_complex(tree) == tree


@pytest.mark.parametrize('data', [
    1j, [1, 1j], [[1j, 2], {'key': 3j}], {'re': 'a', 'im': 'b'},
    {'factory': 'cat.Ob', 'name': 'x'}])
def test_dumps_complex_data(data):
    """ JSON has no complex numbers, so ``dumps`` encodes them as a pair of
    reals and ``loads`` reads them back, leaving every other datum alone. """
    box = cat.Box('f', Ob('x'), Ob('y'), data=data)
    assert loads(dumps(box)) == box


def test_dumps_complex_gate():
    """ Every quantum gate carries a complex array, so none of them used to
    serialise at all (discopy/discopy#775). """
    from discopy.quantum.gates import H, X, Y, Z, S, T
    for gate in (H, X, Y, Z, S, T, H >> X, H @ X):
        assert loads(dumps(gate)) == gate

def test_named_generic_pickle_roundtrip():
    from discopy.frobenius import Diagram, Ty
    from discopy.matrix import Matrix
    from discopy.monoidal import List
    from discopy.quantum import Circuit, Ket, qubit
    from discopy.tensor import Dim, Tensor
    for x in (Tensor[float]([1.0], dom=Dim(1), cod=Dim(1)),
              Matrix[float]([1.0], 1, 1),
              List[int]((1, 2)), List[type]((int, )),
              Box("f", Dim(1), Dim(1), data=[1.0]),
              Diagram.id(Ty('x')).to_hypergraph(),
              Ket(0) >> Circuit.id(qubit)):
        y = pickle.loads(pickle.dumps(x))
        assert type(y) is type(x) and y == x
        assert "__class_getitem__values__" not in y.__dict__


def test_named_generic_setstate_reads_an_old_pickle():
    from discopy.tensor import Dim
    box, state = Box.__new__(Box), {
        "z": 0, "name": "f", "data": None, "is_dagger": False,
        "dom": Dim(1), "cod": Dim(1), "inside": (),
        "__class_getitem__values__": (float, )}
    box.__setstate__(state)
    assert type(box) is Box[float] and box.dtype is float
    assert "__class_getitem__values__" not in box.__dict__

def test_resolve_factory_dtype():
    import numpy as np
    from discopy import cat, tensor
    assert resolve_factory('discopy.cat.Box') == cat.Box
    assert resolve_factory('float') == float
    assert resolve_factory('float64') == np.float64
    assert resolve_factory('tensor.Box[float]') == tensor.Box[float]
    assert resolve_factory('tensor.Box[float64]') == tensor.Box[np.float64]
    with pytest.raises(AttributeError):
        resolve_factory('tensor.Box[NotAType]')
    with pytest.raises(AttributeError):
        resolve_factory('pi')


def test_tensor_tree_roundtrip():
    # No tensor box round-tripped through its tree, see #781.
    from discopy.tensor import Dim
    box = Box('f', Dim(2), Dim(3), [[1., 2., 3.], [4., 5., 6.]])
    for x in (box, box >> box.dagger(), Box('g', Dim(1), Dim(1), [2.])):
        assert loads(dumps(x)) == x
    # A dtype asked for explicitly is not the one the array implies.
    subscripted = Box[complex]('f', Dim(1), Dim(1), [1.])
    assert loads(dumps(subscripted)) == subscripted\
        != Box('f', Dim(1), Dim(1), [1.])
def test_unbiased_keyword_operand():
    """ An operand passed by keyword is composed, not dropped. """
    from discopy.cat import Functor
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    F, G = Functor({x: y}, {}), Functor({y: z}, {})
    assert F.then(other=G)(x) == F.then(G)(x) == z
    assert F.then() == F
