import pickle

import pytest

from unittest.mock import MagicMock
from unittest.mock import patch

from discopy.cat import Ob
from discopy.utils import *

zip_mock = MagicMock()
zip_mock.open().__enter__().read.return_value =\
    '[{"factory": "cat.Ob", "name": "a"}]'


@patch('urllib.request.urlretrieve', return_value=(None, None))
@patch('zipfile.ZipFile', return_value=zip_mock)
def test_load_corpus(a, b):
    assert load_corpus("[fake url]") == [Ob("a")]


def test_named_generic_cache():
    from discopy import tensor as dt
    box, box_int, box_float = dt.Box, dt.Box[int], dt.Box[float]
    assert box_int is dt.Box[int]
    assert box is not box_int and box_float is not box_int
    diag_int = dt.Diagram[int]
    assert diag_int is dt.Diagram[int]
    assert box_int is dt.Box[int]


def test_parameterised_pickle_and_deepcopy():
    from copy import deepcopy
    from discopy import frobenius, interaction, matrix, symmetric
    x = frobenius.Ty('x')
    f = frobenius.Box('f', x, x @ x)
    for original in (matrix.Matrix[int]([0, 1, 1, 0], 2, 2),
                     interaction.Ty[symmetric.Ty](symmetric.Ty('a')),
                     (f >> f @ x).to_hypergraph()):
        for copy in (pickle.loads(pickle.dumps(original)), deepcopy(original)):
            assert type(copy) is type(original) and copy == original
            assert '__class_getitem__values__' not in vars(copy)


def test_named_generic_needs_a_parameter():
    class Plain(NamedGeneric):
        pass
    with pytest.raises(TypeError, match="no type parameter"):
        Plain[int]


def test_MappingOrCallable():
    from typing import Callable, Mapping
    f, g = MappingOrCallable(lambda x: x + 1), MappingOrCallable({0: 1})
    assert f[0] == g[0] == 1 and len(g) == 1 and list(g) == [0]
    assert MappingOrCallable[int, int]\
        == Mapping[int, int] | Callable[[int], int]


def test_inductive():
    from discopy import stream, symmetric
    x = stream.Ty.sequence(symmetric.Ty('x'))
    with pytest.raises(ValueError):
        x.delay(-1)


def test_parameterised_factory_name():
    assert from_tree({'factory': 'cat.Ob[int]', 'name': 'x'}) == Ob('x')


def test_generator():
    from discopy import markov, closed, compact, feedback
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
    alias = vars(symmetric.Diagram)["Braid"]
    assert alias.root is None and alias.parents == ()
    with pytest.raises(TypeError, match="no class to build"):
        alias.build(symmetric.Diagram)
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
