from os import listdir

import pickle
import re

import pytest
from pytest import warns

from unittest.mock import MagicMock
from unittest.mock import patch

from discopy import rigid
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


def test_generator():
    from discopy import symmetric, markov, closed, compact, feedback
    from discopy import biclosed, rigid, pivotal
    from discopy.grammar import categorial
    assert closed.Wire.__bases__ == (biclosed.Wire, )
    assert categorial.Over.ob is categorial.Ty
    assert pivotal.Functor.dom is pivotal.Functor.cod is pivotal.Diagram
    assert rigid.Nat.Exp is rigid.Exp
    assert closed.Swap.__bases__ == (
        markov.Swap, closed.Permutation, closed.Box, closed.Diagram)
    assert closed.Discard.__bases__ == (
        markov.Discard, closed.Copy, closed.Diagram)
    assert pickle.loads(pickle.dumps(closed.Swap)) is closed.Swap
    x, y = compact.Ty('x'), feedback.Ty('y')
    assert compact.Swap(x, x).r == compact.Swap(x.r, x.r)
    assert feedback.Swap(y, y).delay().dom == y.delay() @ y.delay()


def test_generator_decorator():
    """ A category binds a generator under its own name, and is inherited. """
    from discopy import cat

    @factory
    class Base(cat.Arrow):
        pass

    @Base.generator
    class Atom(cat.Box, Base):
        pass

    @factory
    class Sub(Base):
        pass

    assert Base.Atom is Atom is Sub.Atom


def test_Generator_alias():
    """ A factory can be another factory of the same category. """
    from discopy import symmetric, closed
    assert symmetric.Diagram.Braid is symmetric.Swap
    assert closed.Ty.Over is closed.Ty.Under is closed.Exp


def test_Generator_outside_the_hierarchy():
    """ A factory class with no generator in its bases keeps the root. """
    from discopy import monoidal
    from discopy.grammar import cfg
    assert cfg.Rule.Layer is monoidal.Layer
    assert cfg.Word.Box is monoidal.Box


def test_Generator_call():
    """ A factory taken out of its class calls the generator of its owner. """
    from discopy import symmetric
    x = symmetric.Ty('x')
    swap = vars(symmetric.Diagram)["Swap"]
    twist = vars(symmetric.Diagram)["Twist"]
    assert swap(x, x) == symmetric.Swap(x, x)
    assert twist(x) == symmetric.Diagram.id(x)


def test_generator_override():
    """ A generator declared or assigned by hand wins over a built one. """
    from discopy import monoidal, symmetric, tensor

    @factory
    class Recipe(symmetric.Diagram):
        pass

    @Recipe.generator
    class Box(symmetric.Box, Recipe):
        pass

    assert Recipe.Swap.__bases__ == (
        symmetric.Swap, Recipe.Permutation, Box, Recipe)
    assert monoidal.Nat.Wire is monoidal.Dim.Wire is int
    assert tensor.Diagram[complex].Swap is tensor.Swap


@pytest.mark.parametrize("path", [
    "braided.Diagram", "traced.Diagram", "balanced.Diagram",
    "symmetric.Diagram", "markov.Diagram", "closed.Diagram",
    "biclosed.Diagram", "rigid.Diagram", "pivotal.Diagram", "ribbon.Diagram",
    "compact.Diagram", "frobenius.Diagram", "feedback.Diagram",
    "tensor.Diagram", "grammar.pregroup.Diagram", "grammar.categorial.Diagram",
    "quantum.circuit.Circuit", "quantum.zx.Diagram"])
def test_generator_exports(path):
    """ Every generator of a level is exported by the module defining it. """
    import sys
    from importlib import import_module
    module, name = path.rsplit(".", 1)
    D = getattr(import_module(f"discopy.{module}"), name)
    for owner in (D, D.ob):
        names = {name for klass in owner.__mro__ for name, value
                 in vars(klass).items() if isinstance(value, Generator)}
        assert names
        for name in names:
            if isinstance(cls := getattr(owner, name), type):
                assert getattr(sys.modules[cls.__module__], cls.__name__)\
                    is cls
