# -*- coding: utf-8 -*-

import inspect
import sys
from functools import cache

import pytest
from pytest import raises

from discopy.cat import *
from discopy.utils import AxiomError


def test_main():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
    assert Id(x) >> f == f == f >> Id(y)
    assert (f >> g).dom == f.dom and (f >> g).cod == g.cod
    assert f >> g >> h == f >> (g >> h)
    F = Functor(ob_map={x: y, y: z, z: x}, ar_map={f: g, g: h})
    assert F(Id(x)) == Id(F(x))
    assert F(f >> g) == F(f) >> F(g)


def test_Ob():
    assert Ob('x') == Ob('x') and Ob('x') != Ob('y')


def test_Ob_init():
    assert (Ob('x'), Ob('Alice')) == (Ob('x'), Ob('Alice'))


def test_Ob_name():
    assert Ob('x').name == 'x'


def test_Ob_repr():
    assert repr(Ob('x')) == "cat.Ob('x')"


def test_Ob_str():
    assert str(Ob('x')) == 'x'


def test_Ob_eq():
    x, x1, y = Ob('x'), Ob('x'), Ob('y')
    assert x == x1 and x != y and x != 'x'
    assert 'x' != Ob('x')


def test_Ob_hash():
    assert {Ob('x'): 42}[Ob('x')] == 42


def test_Arrow():
    x, y, z, w = Ob('x'), Ob('y'), Ob('z'), Ob('w')
    f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, w)
    assert f >> g >> h == Arrow((f, g, h), x, w)


def test_Arrow_init():
    with raises(TypeError) as err:
        Arrow((), 1, Ob('x'))
    with raises(TypeError) as err:
        Arrow((), Ob('x'), 1)
    with raises(TypeError) as err:
        Arrow((Ob('x'), ), Ob('x'), Ob('x'))
    with raises(TypeError):
        Arrow((), None, None)  # Only a type has None, i.e. no colour.


def test_Arrow_init_iterator():
    x, y = Ob('x'), Ob('y')
    f, g = Box('f', x, x), Box('g', y, y)
    assert Arrow(iter([f, f]), x, x) == f >> f
    with raises(AxiomError):
        Arrow(iter([f, g]), x, x)
    with raises(TypeError):
        Arrow(iter([x]), x, x)


def test_Arrow_len():
    assert len(Arrow((), Ob('x'), Ob('x'))) == 0


def test_Arrow_getitem():
    f, g = Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z'))
    arrow = f >> g >> g.dagger() >> f.dagger()\
        >> f >> g >> g.dagger() >> f.dagger()
    with raises(TypeError):
        arrow["Alice"]
    with raises(IndexError):
        arrow[9]
    assert arrow[:] == arrow
    assert arrow[::-1] == arrow.dagger()
    assert arrow[:0] == arrow[:-8] == arrow[-9:-9] == Id(arrow.dom)
    # Non-unit steps return the corresponding sub-path (daggered when the
    # step is negative), raising only when the result is not composable.
    h = Box('h', Ob('x'), Ob('x'))
    loop = h >> h >> h
    assert loop[::2] == h >> h
    assert loop[::-2] == (h >> h).dagger()
    with raises(AxiomError):
        arrow[::2]
    with raises(AxiomError):
        arrow[::-2]
    for depth, box in enumerate(arrow):
        assert arrow[depth] == box
        assert arrow[-depth] == arrow.inside[-depth]
        assert arrow[depth:depth] == Id(box.dom)
        assert arrow[depth:] == Id(box.dom).then(*arrow.inside[depth:])
        assert arrow[:depth] == Id(arrow.dom).then(
            *arrow.inside[:depth])
        assert arrow[depth: depth + 2] == Id(box.dom).then(
            *arrow.inside[depth: depth + 2])


def test_Arrow_repr():
    assert repr(Arrow((), Ob('x'), Ob('x'))) == "cat.Arrow.id(cat.Ob('x'))"
    inside = (Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z')))
    assert repr(Arrow(inside, Ob('x'), Ob('z')))\
        == "cat.Arrow(inside=(cat.Box('f', cat.Ob('x'), cat.Ob('y')), "\
           "cat.Box('g', cat.Ob('y'), cat.Ob('z'))), dom=cat.Ob('x'), "\
           "cod=cat.Ob('z'))"


def test_Arrow_str():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert str(Arrow((), x, x)) == "Id(x)"
    assert str(Arrow((f, ), x, y)) == "f"
    assert str(Arrow((f, g), x, z)) == "f >> g"


def test_Arrow_eq():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert f >> g == Arrow((f, g), x, z)


def test_Arrow_hash():
    assert {Id(Ob('x')): 42}[Id(Ob('x'))] == 42


def test_Arrow_then():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    assert f.then(g) == f >> g == g << f
    with raises(TypeError) as err:
        f >> x


def test_Arrow_dagger():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    h = Arrow((f, g), x, z)
    assert h.dagger() == g.dagger() >> f.dagger()
    assert h.dagger().dagger() == h
    assert isinstance(h, DaggerCategory)
    assert not issubclass(FreeCategory, DaggerCategory)
    assert not issubclass(Functor, DaggerCategory)


def test_Id_init():
    idx = Id(Ob('x'))
    assert idx >> idx == idx
    assert idx.dagger() == idx


def test_Id_repr():
    assert repr(Id(Ob('x'))) == "cat.Arrow.id(cat.Ob('x'))"


def test_Id_str():
    x = Ob('x')
    assert str(Id(x)) == "Id(x)"


def test_AxiomError():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    with raises(AxiomError) as err:
        Arrow((g, ), x, y)
    with raises(AxiomError) as err:
        Arrow((f, ), x, z)
    with raises(AxiomError) as err:
        g >> f


def test_Box():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
    assert f >> Id(Ob('y')) == f == Id(Ob('x')) >> f


def test_Box_dagger():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
    assert f.dom == f.dagger().cod and f.cod == f.dagger().dom
    assert f == f.dagger().dagger()


def test_Box_subclass_rebuild():
    class Gate(Box):
        def __init__(self, label, source, target, **params):
            super().__init__(label, source, target, **params)
    gate = Gate('g', Ob('x'), Ob('y'), data=42)
    assert gate.dagger().dagger() == gate and gate.rebuild(data=1).data == 1


def test_Bubble_dagger():
    f = Box('f', Ob('x'), Ob('y'))
    b = f.bubble(data=42)
    assert b.dagger().data == 42 and b.dagger().is_dagger
    assert b.dagger().dagger() == b


def test_Bubble_data_eq():
    f = Box('f', Ob('x'), Ob('y'))
    one, two = f.bubble(data=1), f.bubble(data=2)
    assert one != two and len({one, two, f.bubble(data=1)}) == 2


def test_Box_repr():
    f = Box('f', Ob('x'), Ob('y'), data=42)
    assert repr(f) == "cat.Box('f', cat.Ob('x'), cat.Ob('y'), data=42)"
    assert repr(f.dagger())\
        == "cat.Box('f', cat.Ob('x'), cat.Ob('y'), data=42).dagger()"


def test_Box_str():
    f = Box('f', Ob('x'), Ob('y'), data=42)
    assert str(f) == "f"
    assert str(f.dagger()) == "f[::-1]"


def test_Box_hash():
    assert {Box('f', Ob('x'), Ob('y')): 42}[Box('f', Ob('x'), Ob('y'))] == 42


def test_Box_eq():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
    assert f == Arrow((f, ), Ob('x'), Ob('y')) and f != Ob('x')


def test_Box_generator_hash():
    """
    A box and the length-one arrow made of that box are equal, so they must
    also hash equally, see https://github.com/discopy/discopy/pull/387
    """
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    assert f.is_generator and f.generator is f
    arrow = Id(x) >> f
    assert arrow.is_generator and arrow.generator == f
    assert f == arrow and hash(f) == hash(arrow)
    assert {f: 42}[arrow] == 42 and {arrow: 42}[f] == 42
    # A genuine arrow of length two is not a generator.
    g = Box('g', y, x)
    assert not (f >> g).is_generator and (f >> g).generator is None


def test_Sum_generator_hash():
    """
    A singleton ``Sum`` equals its unique term, so it must hash like it too,
    see https://github.com/discopy/discopy/pull/387
    """
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    singleton = Sum((f, ), x, y)
    assert singleton == f and hash(singleton) == hash(f)
    assert {f: 42}[singleton] == 42


def test_Sum_generator_is_its_only_term():
    x, y = Ob('x'), Ob('y')
    f, g = Box('f', x, y), Box('g', x, y)
    singleton = Sum((f, ))
    assert singleton.is_generator and singleton.generator is f
    assert singleton.generator.setoid() == singleton.setoid()
    assert (f + g).generator is None and Sum((), x, y).generator is None


def test_Functor():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    F = Functor({x: y, y: x, z: z}, {f: f.dagger(), g: f >> g})
    assert F((f >> g).dagger()) == F(f >> g).dagger()
    assert F(Id(Ob('x'))) == Id(Ob('y'))


def test_Functor_eq():
    x, y = Ob('x'), Ob('y')
    assert Functor({x: y, y: x}, {}) == Functor({y: x, x: y}, {})
    F, G = Functor({}, {}), Functor({}, {}, dom=Box)
    assert F != G
    assert repr(G) == "cat.Functor(ob_map={}, ar_map={}, dom=cat.Box)"


def test_Functor_repr():
    assert repr(Functor({}, {})) == "cat.Functor(ob_map={}, ar_map={})"


def test_Functor_call():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g = Box('f', x, y), Box('g', y, z)
    F = Functor({x: y, y: x, z: z}, {f: f.dagger(), g: f >> g})
    with raises(TypeError) as err:
        F(F)
    assert F(x) == y
    assert F(f) == f.dagger()
    assert F(f.dagger()) == f
    assert F(g) == f >> g
    assert F(f >> g) == f.dagger() >> f >> g


def test_Functor_into_a_category_with_no_dagger():
    """ A daggered box is read off `ar_map` when the codomain has no dagger,
    and stays the dagger of the image when it has one. """
    from discopy.python import multiplicative
    x = Ob('x')
    f = Box('f', x, x)
    F = Functor(
        {x: int}, lambda g: (lambda n: n - 1) if g.is_dagger
        else (lambda n: n + 1), cod=multiplicative.Function)
    assert F(f)(1) == 2 and F(f.dagger())(1) == 0
    assert Functor({x: x}, {f: f})(f.dagger()) == f.dagger()



def test_Functor_dagger_is_declared():
    """ rel-int/discopy-next#373: a daggered box is the dagger of its image
    exactly when the codomain declares a dagger category. """
    from discopy import monoidal
    from discopy.abc import DaggerCategory
    from discopy.drawing import Drawing
    from discopy.python import additive
    assert issubclass(Drawing, DaggerCategory)
    assert not issubclass(additive.Function, DaggerCategory)
    assert not hasattr(additive.Function, "dagger")
    x = monoidal.Ty('x')
    f = monoidal.Box('f', x, x)
    F = monoidal.Functor(
        {x: int}, lambda g: (lambda n: n - 1) if g.is_dagger
        else (lambda n: n + 1), cod=additive.Function)
    assert F(f)(1) == 2 and F(f.dagger())(1) == 0
    G = monoidal.Functor({x: x}, {f: f.to_drawing()}, cod=Drawing)
    assert G(f.dagger()) == f.to_drawing().dagger()


def test_Transformation():
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    F, G = Functor.id(), Functor({x: y, y: x}, {})
    alpha = Transformation({x: f, y: f[::-1]}, F, G)
    assert alpha(x) == f and alpha(y) == f[::-1]
    assert alpha.dom == F and alpha.cod == G


def test_Transformation_id():
    x, y = Ob('x'), Ob('y')
    F = Functor({x: y, y: x}, {})
    idF = Transformation.id(F)
    assert idF.dom == idF.cod == F
    assert idF(x) == Id(F(x)) == Id(y)


def test_Transformation_then():
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    F, G = Functor.id(), Functor({x: y, y: x}, {})
    alpha = Transformation({x: f, y: f[::-1]}, F, G)
    beta = Transformation.id(G)
    assert (alpha >> beta)(x) == alpha(x) >> beta(x)
    assert (alpha >> beta).dom == F and (alpha >> beta).cod == G


def test_Transformation_laws():
    """ The identity is shared and dropped on either side, and dictionaries
    compose into dictionaries, so the laws of a category hold up to ==. """
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    F, G = Functor.id(), Functor({x: y, y: x}, {})
    alpha = Transformation({x: f, y: f[::-1]}, F, G)
    beta = Transformation({x: f[::-1], y: f}, G, F)
    gamma = Transformation(lambda z: f if z == x else f[::-1], F, G)
    idF, idG = Transformation.id(F), Transformation.id(G)
    assert idF == Transformation.id(F)
    for delta in (alpha, gamma):
        assert idF >> delta == delta == delta >> idG
    assert alpha >> beta == alpha >> beta
    assert (alpha >> beta).components == {x: f >> f[::-1], y: f[::-1] >> f}
    assert (alpha >> beta) >> alpha == alpha >> (beta >> alpha)
    assert gamma >> beta != gamma >> beta


def test_Transformation_eq():
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    F, G = Functor.id(), Functor({x: y, y: x}, {})
    assert Transformation({x: f, y: f[::-1]}, F, G)\
        == Transformation({x: f, y: f[::-1]}, F, G)
    assert Transformation({x: f, y: f[::-1]}, F, G) != F


def test_Transformation_repr():
    F = Functor.id()
    assert "Transformation" in repr(Transformation.id(F))


def test_Transformation_errors():
    x, y = Ob('x'), Ob('y')
    F = Functor.id()
    H = Functor({x: y, y: x}, {})
    # Vertical composition requires self.cod == other.dom.
    with raises(AxiomError):
        Transformation.id(F) >> Transformation.id(H)
    # then only composes with another Transformation.
    with raises(TypeError):
        Transformation.id(F) >> F
    # A component must be an arrow from dom(x) to cod(x).
    f = Box('f', x, y)
    with raises(AxiomError):
        Transformation({x: f, y: f[::-1]}, F, F)(x)


def test_total_ordering():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    assert sorted([z, y, x]) == [x, y, z]
    f, g = Box('f', x, y), Box('g', y, z)
    assert f < g


def test_Bubble():
    f = Box('f', Ob('x'), Ob('y'))
    assert repr((f).bubble())\
        == "cat.Box('', cat.Ob('x'), cat.Ob('y'), "\
        "args=(cat.Box('f', cat.Ob('x'), cat.Ob('y')),))"
    assert str(f.bubble()) == "(f).bubble()"
    named = f.bubble(name="N", method="m")
    assert repr(named) == "cat.Box('N', cat.Ob('x'), cat.Ob('y'), "\
        "args=(cat.Box('f', cat.Ob('x'), cat.Ob('y')),), method='m')"
    assert eval(repr(named), {"cat": __import__("discopy").cat}) == named
    assert Functor.id()(f.bubble(name="N")) == f.bubble(name="N")
    image = Functor.id()(f.bubble(data=42, is_dagger=True))
    assert (image.data, image.is_dagger) == (42, True)
    empty = Bubble(dom=Ob('x'), cod=Ob('y'))
    assert Functor.id()(empty) == empty


def test_Bubble_str():
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    shapes = (f.bubble(name="N"), f.bubble(method="m"),
              f.bubble(dom=y, cod=x, name="N", method="m"),
              f.bubble(data=1), f.bubble(name="N", data=[1, 2]))
    for bubble in shapes + tuple(shape.dagger() for shape in shapes):
        assert eval(str(bubble)) == bubble


def test_Box_call():
    f = Box('f', Ob('x'), Ob('y'))
    with raises(TypeError):
        f(42)


def test_from_tree():
    f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
    d = (f >> f[::-1].bubble()) + Id(Ob('x'))
    assert from_tree(d.to_tree()) == d


def test_sum_lambdify():
    pytest.importorskip("sympy")
    from sympy.abc import phi
    f = Box('f', Ob('x'), Ob('y'), data=[phi])
    g = Box('g', Ob('x'), Ob('y'), data=[phi])

    assert (f + g).free_symbols == {phi}
    assert (f + g).lambdify(phi)(1) == f.lambdify(phi)(1) + g.lambdify(phi)(1)

    empty_sum = Sum((), Ob('x'), Ob('y'))
    assert empty_sum.lambdify(phi)(123) == empty_sum

    assert empty_sum.subs(phi, 0) == empty_sum


def test_Bubble_subs_lambdify():
    pytest.importorskip("sympy")
    from sympy.abc import phi, psi
    f = Box('f', Ob('x'), Ob('x'), data=phi)
    bubble = Bubble(f, f, name="b", method="m", data=phi)
    expected = Bubble(f.subs(phi, 1), f.subs(phi, 1),
                      name="b", method="m", data=1)
    assert bubble.subs(phi, 1) == bubble.lambdify(phi)(1) == expected
    assert bubble.subs(phi, 1).data == bubble.lambdify(phi)(1).data == 1
    assert bubble.subs(psi, 1) == bubble.lambdify(psi)(1) == bubble


def test_subs_forms_and_data():
    pytest.importorskip("sympy")
    import numpy as np
    from sympy.abc import phi, psi
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y, data=phi + psi)
    expected = Box('f', x, y, data=3)
    assert f.subs(phi, 1).data == f.subs([(phi, 1)]).data\
        == f.subs((phi, 1)).data == psi + 1
    for args in [([(phi, 1), (psi, 2)], ), ({phi: 1, psi: 2}, ),
                 ((phi, 1), (psi, 2))]:
        assert f.subs(*args) == expected
    g = Box('g', x, y, data=["phi", {"phase": phi}])
    assert g.free_symbols == {phi}
    assert Box('h', x, y, data="h").free_symbols == set()
    assert g.subs(phi, 1).data == g.lambdify(phi)(1).data\
        == ["phi", {"phase": 1}]
    h = Box('h', x, y, data=np.array([[phi, 1], [psi, 2]], dtype=object))
    assert np.array_equal(h.subs(psi, 4).subs(phi, 3).data, [[3, 1], [4, 2]])
    import sympy
    for data in (h.data, sympy.Matrix(h.data), sympy.Array(h.data)):
        h = Box('h', x, y, data=data)
        for result in (h.subs([(phi, 3), (psi, 4)]), h.lambdify(phi, psi)(3, 4)):
            assert np.array_equal(np.array(result.data), [[3, 1], [4, 2]])
    empty = Box('e', x, y, data=np.zeros((0, 3))).subs(phi, 1).data
    assert empty.shape == (0, 3)
    lam = sympy.Symbol("lambda")
    for data in (sympy.sin(phi) + sympy.sin(psi), phi + lam):
        assert Box('p', x, y, data=data).subs(phi, 1).data == data.subs(phi, 1)
    from discopy.utils import rsubs
    data = [sympy.sin(phi) + psi, phi, sympy.Matrix([[sympy.sin(phi)]])]
    assert rsubs(data, sympy.sin(phi), 0) == rsubs(data, {sympy.sin(phi): 0})\
        == [psi, phi, sympy.Matrix([[0]])]


def test_subs_key_not_a_symbol():
    sympy = pytest.importorskip("sympy")
    from sympy.abc import phi, psi
    x, sin = Ob('x'), sympy.sin(phi)
    f, g = Box('f', x, x, data=sin + psi), Box('g', x, x, data=phi + psi)
    assert f.subs(sin, 0).data == psi and g.subs(phi + psi, 0).data == 0
    assert f.subs({sin: 0, psi: 1}).data == 1
    assert (f >> g).subs(sin, 0) == f.subs(sin, 0) >> g
    bubble = Bubble(f, data=sin).subs(sin, 0)
    assert bubble == Bubble(f.subs(sin, 0), data=0) and bubble.data == 0


def test_free_symbols_of_numeric_array_read_from_dtype():
    import numpy as np

    class Opaque(np.ndarray):
        def __iter__(self):
            raise AssertionError("a numeric array was iterated")

    data = np.ones((2, 2)).view(Opaque)
    assert Box('f', Ob('x'), Ob('x'), data=data).free_symbols == set()


def test_free_symbols_follow_mutable_data():
    pytest.importorskip("sympy")
    from sympy.abc import psi
    f = Box('f', Ob('x'), Ob('x'), data=[0])
    assert f.free_symbols == set()
    f.data.append(psi)
    assert f.free_symbols == {psi}
    assert f.subs(psi, 1).data == f.lambdify(psi)(1).data == [0, 1]


def test_Sum():
    x, y, z = map(Ob, "xyz")
    f = Box('f', x, y)
    g = Box('g', y, z)
    with raises(ValueError):
        Sum(())
    with raises(AxiomError):
        Sum((), x, y) + Sum((), y, z)
    with raises(AxiomError):
        Sum((f, ), y, z)
    with raises(AxiomError):
        Sum((f, g))
    assert hash(Sum((), x, y)) == hash(Sum((), x, y))
    assert repr(Sum((), x, y))\
        == "cat.Sum(terms=(), dom=cat.Ob('x'), cod=cat.Ob('y'))"
    lazy = Sum((f, f))
    assert "name" not in vars(lazy) and lazy.name == repr(lazy)\
        == f"cat.Sum(terms=({f!r}, {f!r}))"
    assert list(Sum((f, ), x, y)) == [f]
    assert len(Sum((), x, y)) == 0
    assert Sum((), x, x).then(f, g) == Sum((), x, z)
    assert Sum((), x, y).dagger() == Sum((), y, x)
    h = Box('h', z, x)
    with raises(AxiomError):
        Sum((), x, y) >> h
    with raises(AxiomError):
        h >> Sum((), y, z)


def test_Sum_terms_is_a_tuple():
    x, y = Ob('x'), Ob('y')
    f, g = Box('f', x, y), Box('g', x, y)
    assert Sum([f, g]) == Sum(iter([f, g])) == Sum((f, g)) == f + g
    assert hash(Sum([], x, y)) == hash(Sum((), x, y))


def test_Functor_then_unit():
    """ An identity is dropped on either side of a composite (#142). """
    x, y = Ob('x'), Ob('y')
    F, G = Functor({x: y, y: x}, {}), Functor(lambda x: x, lambda f: f[::-1])
    for H in (F, G):
        assert Functor.id() >> H == H == H >> Functor.id()
    assert Functor.id() == Functor.id() >> Functor.id()
    assert G >> G != G >> G
    alpha, beta = (Transformation.id(Functor.id()) for _ in range(2))
    assert (alpha >> beta)(x) == Arrow.id(x)

    class Strict:
        """ A map whose equality assumes its own type. """
        key = 0

        def __call__(self, x):
            return x

        def __eq__(self, other):
            return self.key == other.key

    H = Functor(Strict(), Strict())
    assert (H >> F)(x) == F(x) == (F >> H)(x)


def test_Functor_then_applies_the_first_functor():
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    G = Functor({x: x, y: y}, {f: f})
    for F in (Functor({x: 'y', y: 'x'}, {f: (f[::-1], )}),
              Functor(lambda ob: ob.name, lambda box: (box, ))):
        assert (F >> G)(x) == G(F(x)) and (F >> G)(f) == G(F(f))


def test_Bubble_no_common_boundary():
    x, y = Ob('x'), Ob('y')
    f, g = Box('f', x, x), Box('g', y, x)
    with raises(ValueError, match="needs a dom given"):
        Bubble(f, g)
    with raises(ValueError, match="needs a cod given"):
        Bubble(f, g[::-1], dom=x)
    assert Bubble(f, g, dom=x).cod == x
    with raises(ValueError, match=r"they have \[\]"):
        Bubble()


ABSTRACT = ("Timed", "Structural", "ControlledRotation", "BinaryTerm", "Parametrized")


@cache
def box_classes():
    """
    Every class of box that the package defines, see #169, but for those of
    :mod:`discopy.optyx`, vendored with constructors of its own.
    """
    import importlib
    import pkgutil
    import discopy
    classes = {}
    for info in pkgutil.walk_packages(discopy.__path__, "discopy."):
        if info.name.startswith("discopy.optyx"):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # it needs an extra, see test/plugin.py
            continue
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, Box)\
                    and value.__module__ == info.name:
                classes[value] = None
    return list(classes)


EXAMPLES = {  # a box of each constructor, from a class c, types x, y, data v
    "cat.Box": 'c("f", x, y, data=v, is_dagger=True)',
    "{cat,markov}.Sum": "c((box(c, x, y, v), ))",
    "{monoidal,neural.network}.Box": 'c("f", x, y, data=v)',
    "peirce.alpha.Box": 'c("f", data=v)',
    "peirce.beta.Box": 'c("f", x, y, data=v, is_deterministic=True)',
    "peirce.{alpha,beta}.Cut": 'c(c.ar.box_factory("f", t := c.ob(c.ob.generator_'
    'factory("x", dom=(k := sys.modules["discopy.peirce.alpha"].black), '
    'cod=k)), t, data=v), name="b")',  # not the unit, #636
    "monoidal.Layer":
    'c(x, sys.modules[c.__module__].Box("f", x, y, data=v), y)',
    "braided.Braid {balanced,ribbon}.DualRailBraid ribbon.DualRailC{u,a}p":
    "c(x, y, is_dagger=True)",
    "balanced.{Twist,DualRailTwist}": "c(x, is_dagger=True)",
    "symmetric.Swap": "c(x, y)",
    "symmetric.Permutation": "c(x @ y @ x, [2, 0, 1])",
    "feedback.Box": 'c("f", x, y, data=v).delay()',
    "feedback.cat.Box": 'c("f", x, y, data=v).delay()',
    "feedback.Feedback": "c(box(c, x @ y.delay(), x @ y, v))",
    "feedback.Feedforward": "c(box(c, x @ y, x @ y.delay(), v))",
    "feedback.Trace": "c(box(c, y @ x, y @ x, v), left=True)",
    "feedback.{Head,Tail}": "c(box(c, x, y, v), 1)",
    "feedback.FollowedBy": "c(x, True, 1)",
    "feedback.Wait feedback.cat.Wait": "c(x)",
    "markov.Box": 'c("f", x, y, data=v, is_deterministic=False)',
    "feedback.{markov,closed}.Box":
    'c("f", x, y, data=v, is_deterministic=False).delay()',
    "markov.Copy comarkov.Merge": "c(x, 3)",
    "markov.Discard comarkov.Noise quantum.gates.{Discard,MixedState} "
    "lambek.cfg.Id": "c(x)",
    "markov.Function comarkov.Cofunction": "c(x @ y, [1, 0, 0])",
    "rigid.Box": 'c("f", x, y, data=v).r',
    "{rigid,frobenius}.Cup": "c(x, x.r)",
    "{rigid,frobenius}.Cap": "c(x, x.l)",
    "pivotal.Box": 'c("f", x, y, data=v).delay().r',
    "feedback.{balanced,symmetric,comarkov}.Box": 'c("f", x, y, data=v).delay()',
    "lambek.pregroup.Box": 'c("f", x, y, data=v, is_deterministic=False).r',
    "tensor.Box": 'c("f", x, y, [v] + [0] * 5)',
    "frobenius.Spider": "c(1, 2, x, data=v)",
    "frobenius.Cospan": "c(x @ y, [0, 0, 1], [1])",
    "linear.{cat,monoidal}.Bang": "c(box(c, x, y, v))",
    "linear.cat.{Dereliction,Digging} "
    "linear.monoidal.{Dereliction,Digging,Contraction,Weakening}": "c(x)",
    "linear.monoidal.Exchange": "c(x, y, True)",
    "linear.monoidal.Laxator": "c(x, y)",
    "closed.monoidal.{Eval,Coeval}": "c(x << y)",
    "closed.symmetric.{Eval,Coeval} closed.markov.{Eval,Coeval}": "c(x << y)",
    "feedback.closed.{monoidal,balanced,symmetric,markov}.{Eval,Coeval}":
    "c(x << y, time_step=1)",
    "closed.{monoidal,symmetric,markov}.Curry":
    "c(box(c, x @ y, x, v), 1, False)",
    "feedback.closed.{monoidal,balanced,symmetric,markov}.Curry":
    "c(box(c, x @ y, x, v), 1, False, time_step=1)",
    "closed.monoidal.Constant": 'c("c", x, data=v)',
    "{closed.markov,lambek.categorial}.Constant "
    "closed.{monoidal,markov}.Variable":
    'c("c", x)',
    "closed.{monoidal,markov}.Application lambek.categorial.FA": "c(f, a)",
    "{closed.monoidal,closed.markov,lambek.categorial}.Abstraction":
    "c(u, f(u))",
    "closed.markov.{Pack,Unpack}": "c(x * y)",
    "feedback.closed.markov.{Pack,Unpack}": "c(x * y, 1)",
    "closed.markov.Tuple": "c(a, u)",
    "closed.markov.Projection": "c(closed.markov.Tuple(a, u), 1)",
    "closed.markov.Let": "c(closed.markov.Tuple(a), [u], u)",
    "lambek.categorial.BA": 'c(a, cg.Constant("f", y >> x))',
    "lambek.categorial.{FTR,BTR}": "c(y, a)",
    "lambek.categorial.TypeRaising": "c(y, a, y << (x >> y))",
    "lambek.categorial.FC": 'c(f, cg.Constant("g", y << x))',
    "lambek.categorial.BC":
    'c(cg.Constant("f", x >> y), cg.Constant("g", y >> x))',
    "lambek.categorial.{FX,BX}": 'c(f, cg.Constant("g", x >> y))',
    "lambek.categorial.ForwardCrossedComposition": "c(x << y, x >> y)",
    "lambek.categorial.BackwardCrossedComposition": "c(y << x, y >> x)",
    "lambek.thue.Rule": 'c(x @ y, x, "r", data=v)',
    "lambek.thue.Word": 'c("w", x @ y, y, data=v)',
    "lambek.cfg.Rule": 'c(x @ y, x, "r")',
    "lambek.cfg.Word": 'c("w", x, y)',
    "lambek.pregroup.Word": 'c("w", x @ y, data=v)',
    "{tensor,quantum.circuit,quantum.zx}.Elementwise":
    'c(box(c, x, x, [v, 0, 0, 1]), func=abs, drawing_name="B")',
    "egraph.Join": "c(box(c, x, y, v), box(c, x, y, v))",
    "quantum.circuit.Box": 'c("f", x, y, data=[v, 0, 0, 1], is_mixed=False)',
    "quantum.gates.QuantumGate": 'c("g", x, x, [v, 0, 0, 1])',
    "quantum.gates.{Ket,Bra}": "c(1, 0, time_step=1)",
    "quantum.gates.Digits": "c(1, 2, dim=3, is_dagger=True, time_step=1)",
    "quantum.gates.{Measure,Encode}": "c(2, False, True, 1)",
    "quantum.gates.{Copy,Match}": "c(1)",
    "quantum.gates.Controlled": "c(gates.Rz(v), -2)",
    "quantum.gates.ControlledRotation": "c(v, -2, 1)",
    "quantum.gates.Rotation": "c(v, 1, 1)",
    "quantum.gates.Scalar": 'c(v, "s", True, 1)',
    "quantum.gates.MixedScalar": "c(v, 1)",
    "quantum.gates.Sqrt": "c(v, False, 1)",
    "quantum.zx.Spider": "c(1, 2, v)",
    "quantum.zx.Scalar": "c(v)",
}


def example(cls, value):
    """ The box of a class that :data:`EXAMPLES` builds, with some data. """
    from discopy import closed, monoidal, tensor
    from discopy.lambek import categorial as cg
    from discopy.neural.network import Dims
    from discopy.quantum import circuit, gates
    from discopy.rigid import Nat
    from discopy.utils import factory_name
    if inspect.isabstract(cls) or cls.__name__ in ABSTRACT:
        return None
    templates = {name: template for key, template in EXAMPLES.items()
                 for name in globs(key)}
    owner = next(c for c in cls.__mro__ if "__init__" in vars(c))
    base = next(c for c in cls.__mro__ if factory_name(c) in templates)
    assert issubclass(base, owner), f"{owner} needs an example"
    ob = sys.modules[cls.__module__].Ty\
        if issubclass(cls, monoidal.Layer) else cls.ob
    x, y = (tensor.Dim(2), tensor.Dim(3)) if issubclass(cls, tensor.Box)\
        else (Dims(2), Dims(3)) if ob is Dims\
        else (Nat(1), Nat(1)) if ob is Nat else (ob('x'), ob('y'))
    if issubclass(cls, circuit.Box):
        x, y = circuit.qubit, circuit.qubit
    terms = getattr(ob, "constant_factory", None) and (
        ob.constant_factory("f", x << y), ob.constant_factory("a", y),
        ob.variable_factory("u", y))
    f, a, u = terms or (None, None, None)
    return eval(templates[factory_name(base)], dict(
        c=cls, x=x, y=y, v=value, f=f, a=a, u=u, sys=sys, closed=closed,
        cg=cg, gates=gates, box=lambda c, x, y, v: c.ar.box_factory(
            "f", x, y, data=v)))


def holds(law, cls):
    """ Whether a law of #169 holds of the example of a class. """
    from discopy.utils import dumps, loads
    box = example(cls, 1)
    if law == "dagger":
        return box.dagger().dagger() == box
    if law == "rotation":
        return not hasattr(box, "rotate") or box.r.l == box == box.l.r
    if law == "rebuild":
        result, expected = box.rebuild(), box
    elif law == "subs":
        from sympy import S
        from sympy.abc import phi
        symbolic, expected = example(cls, phi), example(cls, S.One)
        for result in (symbolic.subs(phi, 1), symbolic.subs([(phi, 1)]),
                       symbolic.lambdify(phi)(1)):
            assert type(result) is type(expected) and result == expected
        return True
    else:
        tree = type(box).from_tree(box.to_tree())
        assert type(tree) is type(box) and tree == box
        result, expected = loads(dumps(box)), box
    return type(result) is type(expected) and result == expected


def globs(patterns: str) -> list[str]:
    """ The globs a pattern stands for, e.g. ``*.{Cup,Cap}``. """
    import itertools
    import re
    result = []
    for pattern in patterns.split():
        parts = re.split(r"\{([^}]*)\}", pattern)
        result += map("".join, itertools.product(*(
            part.split(",") if i % 2 else [part]
            for i, part in enumerate(parts))))
    return result


PIVOTALS = "{pivotal,ribbon,compact,frobenius}"
BROKEN = {
    "rebuild": {
        "*.Layer": "a layer keeps its *inside as boxes_or_types",
        "*.Merge *.Noise": "it keeps neither x nor n",
        "*.Cofunction": "it keeps its cofunc in an op",
        "*.Cospan": "it keeps left and right as legs",
        "*.Spider quantum.zx.[XYZ]": "it keeps no n_legs_in or n_legs_out",
        "{closed.markov,feedback.closed.markov}.{Pack,Unpack}":
        "it is built from its type alone",
    },
    "dagger": {
        "*.{Constant,Variable,Abstraction,Application} "
        "closed.markov.{Tuple,Projection,Let} lambek.categorial.{FA,BA,*TR,"
        "TypeRaising,[FB][CX]}": "a term has no stated dagger",
        "lambek.categorial.*Composition": "a crossed composition has none",
        "*.Curry": "a curry has no dagger",
        "*.{Dereliction,Digging,Contraction,Weakening,Laxator}":
        "!L* has no rule from its codomain back to its domain",
        "{closed.markov,markov,lambek.abstract,feedback.closed.markov,"
        "feedback.markov,linear.markov}"
        ".{Copy,Discard,Function} "
        "{comarkov,feedback.comarkov}.{Cofunction,Merge,Noise}":
        "its category has no dagger",
        "rigid.C[ua]p": "a rigid cup or cap has none, a pivotal one has",
    },
    "rotation": {
        PIVOTALS + ".{Feedback,Feedforward,Head,Tail,FollowedBy,Wait}":
        "it has no stated rotation",
        "*.{Feedback,Feedforward} {feedback,feedback.*,neural.network}.Trace":
        "it raises for a rotation, only a pivotal trace rotates",
        "quantum.zx.Scalar": "it takes no dom, cod or z",
    },
    "subs": {
        "*.Spider quantum.zx.[XYZ]": "it keeps no n_legs_in or n_legs_out",
    },
    "tree": {
        "quantum.gates.C[RU]*": "its tree is that of a controlled gate",
        "lambek.categorial.TypeRaising": "its tree drops its cod",
        "neural.network.{Box,Feedback,Feedforward,Trace,Head,Tail}":
        "a module has none",
        "{tensor,quantum.circuit,quantum.zx}.Elementwise":
        "its tree drops its func, which == compares",
    },
}


@pytest.mark.parametrize("law", BROKEN)
def test_every_box(law):
    """
    The conformance test of #169: every class of box rebuilds from its
    fields, and so does its dagger, its rotations, its substitutions and its
    tree, but for the breakages each law lists with their reasons.
    """
    if law == "subs":
        pytest.importorskip("sympy")
    from fnmatch import fnmatch
    from discopy.utils import factory_name
    patterns = [glob for key in BROKEN[law] for glob in globs(key)]
    unexpected, fixed = {}, []
    for cls in box_classes():
        if example(cls, 1) is None:
            continue
        name = factory_name(cls)
        try:
            failure = None if holds(law, cls) else "not equal"
        except Exception as error:  # pylint: disable=broad-except
            failure = repr(error)
        if any(fnmatch(name, glob) for glob in patterns):
            fixed += [] if failure else [name]
        elif failure:
            unexpected[name] = failure
    assert not unexpected and not fixed, (unexpected, fixed)


@pytest.mark.parametrize(
    "module", ["feedback.symmetric", "feedback", "feedback.markov",
               "frobenius"])
def test_rebuild_strict_init(module):
    """ A box whose constructor takes only some fields gets the others set,
    as main did, see #169. """
    pytest.importorskip("sympy")
    import importlib
    from sympy.abc import phi
    m = importlib.import_module(f"discopy.{module}")

    class Gate(m.Box):
        def __init__(self, name, dom, cod, data=None, is_dagger=False):
            super().__init__(name, dom, cod, data=data, is_dagger=is_dagger)

    x, y = m.Ty('x'), m.Ty('y')
    gate = Gate("g", x, y, data=phi)
    gate.time_step, gate.is_deterministic = 2, False

    def fields(box):
        return (type(box), box.name, box.dom, box.cod, box.data, box.is_dagger,
                box.time_step, getattr(box, "deterministic", False))
    assert fields(gate.dagger()) == (Gate, "g", y, x, phi, True, 2, False)
    assert fields(gate.subs(phi, 1)) == fields(gate.lambdify(phi)(1))\
        == (Gate, "g", x, y, 1, False, 2, False)
    assert gate.dagger().dagger() == gate


def test_rebuild_other_constructor():
    """ A constructor that does not take the fields of a box raises, as on
    main, rather than rebuild from the wrong arguments. """
    from discopy import rigid

    class Digits(rigid.Box):
        def __init__(self, *digits):
            super().__init__(f"Digits{digits}", rigid.Ty(), rigid.Ty("x"))
    with raises(TypeError):
        Digits(1).r


def test_Functor_tree():
    from discopy.utils import dumps, loads
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y, data=(1, 2))
    for F in (Functor(), Functor({x: y, y: x}, {f: f[::-1] >> f}),
              Functor({x: x}, {f: f}, dom=Arrow, cod=Arrow),
              Functor({x: y}, {f: {'array': (1, 2)}})):
        assert loads(dumps(F)) == F
    for ob_map, ar_map, attr in [
            (lambda x: x, {}, "ob_map"), ({}, lambda f: f, "ar_map")]:
        with raises(TypeError, match=f"its {attr} is a Python callable"):
            dumps(Functor(ob_map, ar_map))
    with raises(TypeError, match="A functor has no tree"):
        dumps(Functor.id())
    for F in (Functor({}, {}, cod=int), Functor({int: x}, {})):
        with raises(TypeError, match="not JSON serialisable"):
            dumps(F)
    tree = Functor().to_tree()
    for name in ("int", "ndarray"):
        with raises(AttributeError, match="No class of DisCoPy"):
            Functor.from_tree(tree | {'cod': name})


def test_Equation_tree():
    from discopy.utils import dumps, loads
    x = Ob('x')
    f = Box('f', x, x)
    equation = Equation(f >> f, f, Id(x), symbols=("=", "->"))
    loaded = loads(dumps(equation))
    assert type(loaded) is Equation and bool(loaded) is bool(equation)
    assert (loaded.terms, loaded.symbols) == (equation.terms, ("=", "->"))
    with raises(TypeError, match="its up_to is a Python callable"):
        dumps(Equation(f, f >> f, up_to=len))
    with raises(TypeError, match="Expected str"):
        Equation.from_tree(Equation(f, f).to_tree() | {'symbols': [[1]]})
    with raises(TypeError, match="Expected str"):
        dumps(Equation(f, f, symbols=[1]))


def test_term_sequence():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
    term = (f >> g >> h).to_term()
    assert str(term) == "x(lambda x0: h(g(f(x0))))"
    assert term.eval() == f >> g >> h and term.dom == term.cod == x
    assert term.constants == [Constant(h), Constant(g), Constant(f)]
    assert Id(x).to_term() == x(lambda x0: x0)
    assert Id(x).to_term().eval() == Id(x)
    F = Functor({x: y, y: x, z: z}, {f: Box('g', y, x), g: Box('f', x, z)})
    assert (f >> g).to_term().eval(F) == F(f >> g)
    a = Variable('a', x)
    assert a.dom == x and a.category is Arrow and not a.constants
    assert Constant(f).dom == x and Constant(f)(a).dom == x


def test_term_normal_form():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f, g, h = (Constant(box) for box in (
        Box('f', x, y), Box('g', y, z), Box('h', z, x)))
    a = Variable('a', x)
    statement = let(f(a), lambda b: let(g(b), lambda c: h(c)))
    assert str(statement) == "let(f(a), lambda b: let(g(b), lambda c: h(c)))"
    assert statement.normal_form() == h(g(f(a)))
    assert statement.eval() == f.box >> g.box >> h.box
    closed = y(lambda b: g(b))
    assert closed.normal_form() == closed and closed.eval() == g.box
    assert x(lambda a: closed(f(a))).normal_form()\
        == x(lambda a: g(f(a)))
    assert g.expand() == y(lambda x: g(x)) and a.expand() == a
    k = Constant(Box('k', x, x))
    shadowed = x(lambda a: let(k(a), lambda a: k(a)))
    assert shadowed.normal_form() == x(lambda a: k(k(a)))
    for term in (statement, closed, g, a, h(g(f(a))), shadowed):
        assert TermBase.normal_form_completeness(term)
        assert TermBase.eta(term) and TermBase.reading(term)
        assert loads(dumps(term)) == term
        assert eval(repr(term), TermBase.environment()) == term


def test_term_alpha_and_substitute():
    x, y = Ob('x'), Ob('y')
    f, a, b = Constant(Box('f', x, y)), Variable('a', x), Variable('b', x)
    first, second = x(lambda a: f(a)), x(lambda b: f(b))
    assert first != second and first.alpha_eq(second)
    assert AlphaEquation(first, second) and not AlphaEquation(f(a), f(b))
    assert not first.alpha_eq(f) and not f(a).alpha_eq(a)
    assert not a.alpha_eq(f(a))
    assert not let(a, lambda c: f(c)).alpha_eq(f(a))
    assert f(a).substitute({a: b}) == f(b) and f.substitute({a: b}) == f
    assert first.substitute({a: b}) == first
    assert Variable.fresh("a", x, f(a)) == Variable("a_", x)


def test_term_errors():
    x, y = Ob('x'), Ob('y')
    f, a = Constant(Box('f', x, y)), Variable('a', x)
    with raises(ValueError, match="does not apply"):
        f(f(a))
    with raises(ValueError, match="does not apply"):
        a(a)
    with raises(ValueError, match="does not apply"):
        f(f)
    with raises(ValueError, match="free variable of f.a."):
        Kappa((Variable('b', x), ), f(a))
    with raises(ValueError, match="free variable of f"):
        Kappa((), f)
    with raises(ValueError):
        Kappa((a, ), f(a), 1)
    with raises(TypeError, match="one level above the others"):
        TermBase.highest(Kappa, Application)
    with raises(ValueError, match="Expected 1 variables, got 2"):
        x(lambda a, b: a)
    assert Kappa((a, ), f(a), 0) == Kappa((a, ), f(a))


@pytest.mark.parametrize("level, structure", [
    ("spacial", lambda module, x, y: module.Id(x @ y)),
    ("feedback", lambda module, x, y: module.Id(x @ y)),
    ("braided", lambda module, x, y: module.Braid(y, x)),
    ("balanced", lambda module, x, y: module.Twist(x) @ y),
    ("comarkov", lambda module, x, y: module.Merge(x) @ y)])
def test_terms_of_a_level(level, structure):
    import importlib
    import pickle
    from discopy import monoidal
    module = importlib.import_module(f"discopy.{level}")
    x, y = module.Ty('x'), module.Ty('y')
    wiring = structure(module, x, y)
    diagram = wiring >> module.Box('f', x @ y, x)
    term = diagram.to_term()
    assert type(term) is module.Kappa and term.category is module.Diagram
    assert module.Diagram.term_factory is module.TermBase
    assert issubclass(module.Kappa, monoidal.Kappa)
    assert term.eval() == diagram
    assert len(term.constants) == len(diagram.boxes)
    assert all(type(c) is module.Constant for c in term.constants)
    for copy in (pickle.loads(pickle.dumps(term)), loads(dumps(term))):
        assert type(copy) is module.Kappa and copy == term
    assert type(x(lambda a: a)) is monoidal.Kappa


def test_terms_from_biclosed_up():
    from discopy import closed, compact, symmetric
    x = compact.Ty('x')
    diagram = compact.Swap(x, x) >> compact.Box('f', x @ x, x)
    term = diagram.to_term()
    assert type(term) is symmetric.Kappa and term.eval() == diagram
    assert type(x(lambda a: a)) is closed.monoidal.Abstraction


def test_term_laws():
    from hypothesis import find
    x, y = Ob('x'), Ob('y')
    laws = (
        TermBase.eta, TermBase.reading, TermBase.normal_form_soundness,
        TermBase.normal_form_idempotence, TermBase.normal_form_completeness,
        TermBase.repr_transparency, TermBase.str_transparency,
        TermBase.serialisation)
    shapes = [
        [0, 0, 2, 1, 1, 1, 0, 1, 0], [0, 1, 1], [1, 0, 2, 0, 2, 1],
        [0, 0, 1, 1, 2, 1], [1, 1, 2, 2, 0, 1, 1]]
    for shape in shapes:
        term = TermBase.generate(y, shape, [x, y], "ab")
        assert all(law(term) for law in laws)
        renamed = TermBase.generate(y, shape, [x, y], "c")
        assert TermBase.alpha_renaming((term, renamed))
        assert TermBase.alpha_soundness((term, renamed))
    with raises(NotImplementedError):
        AlphaEquivalence.shapes()
    with raises(NotImplementedError):
        AlphaEquivalence.generate(y, [], [x], "a")
    first, second, _ = find(TermBase.related(), lambda terms: True)
    assert first.alpha_eq(second)
    assert Renamed[TermBase].checked_params() == ()
    assert Canonical[TermBase].checked_params() == ()


def test_legacy_Bubble_tree():
    """ A tree dumped before #560 fused bubbles into boxes still loads. """
    from discopy import monoidal, rigid
    from discopy.utils import from_tree
    ob = {'factory': 'cat.Ob', 'name': 'x'}
    f = {'factory': 'cat.Box', 'name': 'f', 'dom': ob, 'cod': ob}
    tree = {'factory': 'cat.Bubble', 'arg': f, 'dom': ob, 'cod': ob}
    assert from_tree(tree) == Box('f', Ob('x'), Ob('x')).bubble()
    x = rigid.Ty('x')
    g = rigid.Box('g', x, x).bubble(name="b").r
    tree = g.to_tree() | {'factory': 'rigid.Bubble'}
    assert from_tree(tree) == g
    h = monoidal.Box('h', monoidal.Ty('x'), monoidal.Ty('y'))
    tree = h.bubble().to_tree() | {'factory': 'monoidal.Bubble'}
    del tree['name']
    assert from_tree(tree) == h.bubble()
