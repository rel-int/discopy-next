from __future__ import annotations
from pytest import raises

from discopy import cat, comarkov, feedback, frobenius, monoidal
from discopy.python import additive
from discopy.comarkov import *
from discopy.matrix import Matrix
from discopy.utils import AxiomError, dumps, loads


def test_spider_factory():
    x = Ty('x')
    assert Diagram.spider_factory(3, 1, x) == Merge(x, 3)
    for n_legs_in, n_legs_out, phase in [(2, 2, None), (1, 2, None), (2, 1, .5)]:
        with raises(ValueError, match="only spiders of comarkov.Diagram"):
            Diagram.spider_factory(n_legs_in, n_legs_out, x, phase)


def test_Merge_dagger():
    x = Ty('x')
    for box in (Merge(x), Merge(x, 3), Noise(x)):
        with raises(AxiomError, match="not a dagger category"):
            box.dagger()
    x = frobenius.Ty('x')
    assert frobenius.Diagram.merge(x).dagger() == frobenius.Spider(1, 2, x)


def test_Noise():
    x = Ty('x')
    assert Merge(x, 0) == Noise(x) == Noise(x, 0)
    assert isinstance(Noise(x), Noise) and isinstance(Merge(x, n=0), Noise)
    assert Diagram.noise(x) == Noise(x) and Diagram.noise(Ty()) == Id()
    assert Diagram.noise(x @ x) == Noise(x) @ Noise(x)
    with raises(ValueError):
        Noise(x, 2)
    with raises(ValueError):
        Merge(x @ x)


def test_equations():
    x = Ty('x')
    merge, noise = Merge(x), Noise(x)
    assert Equation(noise @ x >> merge, Id(x), x @ noise >> merge)
    assert Equation(merge @ x >> merge, Merge(x, 3), x @ merge >> merge)
    assert Equation(Swap(x, x) >> merge, merge)
    assert Equation(Diagram.merge(x @ x), x @ Swap(x, x) @ x >> merge @ merge)
    assert not Equation(merge >> Box('f', x, x), Box('f', x, x) @ x >> merge)


def test_repr_and_str():
    x = Ty('x')
    assert repr(Merge(x, 3)) == "comarkov.Merge(feedback.Ty(feedback.Wire('x')), 3)"
    assert repr(Noise(x)) == "comarkov.Noise(feedback.Ty(feedback.Wire('x')), 0)"
    assert str(Merge(x)) == "Merge(x)" and str(Noise(x)) == "Merge(x, 0)"
    for box in (Merge(x), Merge(x, 3), Noise(x)):
        assert eval(repr(box), dict(
            cat=cat, monoidal=monoidal, comarkov=comarkov,
            feedback=feedback)) == box


def test_serialisation_and_pickle():
    from pickle import dumps as pickle_dumps, loads as pickle_loads
    x = Ty('x')
    for box in (Merge(x), Merge(x, 3), Noise(x)):
        assert loads(dumps(box)) == box and pickle_loads(pickle_dumps(box)) == box
        assert type(loads(dumps(box))) is type(pickle_loads(pickle_dumps(box)))\
            is type(box)
    diagram = Noise(x) @ x >> Merge(x)
    assert loads(dumps(diagram)) == diagram


def test_Permutation():
    x, y, z = map(Ty, "xyz")
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    assert isinstance(perm, Box) and perm.cod == z @ x @ y
    assert isinstance(perm.inside[0], Layer)
    assert Permutation(x @ y, [1, 0]) == Swap(x, y)
    assert Equation(perm >> perm.dagger(), Id(x @ y @ z))


def test_Functor():
    x = Ty('x')
    f = Box('f', x, x)
    G = Functor(lambda ob: ob, lambda box: box)
    assert G(f @ f >> Merge(x)) == f @ f >> Merge(x)
    F = Functor({x: 1}, {}, cod=Matrix)
    assert F(Merge(x)) == Matrix.merge(1) and F(Noise(x)) == Matrix.noise(1)
    F = Functor({x: int}, {f: lambda n: n + 1}, cod=additive.Function)
    assert F(Noise(x) @ x >> Merge(x))(41) == 41
    assert F(f @ f >> Merge(x))(41, 1) == 42


def test_hypergraph():
    x = Ty('x')
    f, g, h = Box('f', Ty(), x), Box('g', Ty(), x), Box('h', x, x)
    diagram = f @ g >> Merge(x) >> h
    graph = diagram.to_hypergraph()
    assert graph.is_right_monogamous and graph.is_cocausal
    assert not graph.is_left_monogamous and not graph.is_causal
    assert graph.offsets == (0, 1, None)
    assert graph.to_diagram() == diagram.foliation() == Diagram(
        (Layer(f, g), Layer(Merge(x)), Layer(h)), Ty(), x)
    s = Box('s', Ty(), x)
    diagram = f @ g @ s >> Merge(x) @ x
    assert diagram.to_hypergraph().to_diagram() == Diagram(
        (Layer(f, g, s), Layer(Merge(x), x)), Ty(), x @ x)
    assert (Noise(x) >> h).to_hypergraph().to_diagram() == Noise(x) >> h
    decoded = (h @ h >> Merge(x)).to_hypergraph().to_diagram()
    assert Equation(decoded, h @ h >> Merge(x))
    assert decoded.boxes == [h, h, Merge(x)]
    assert Diagram.permutation([1, 0], x @ x).simplify() == Swap(x, x)
    with raises(ValueError, match="only spiders of comarkov.Diagram"):
        Hypergraph.spiders(1, 2, x).to_diagram()


def test_hypergraph_trace():
    x = Ty('x')
    f = Box('f', x @ x, x @ x)
    traced = (Merge(x) @ x >> f).trace()
    graph = traced.to_hypergraph()
    assert graph.is_right_monogamous and not graph.is_cocausal
    assert graph.make_cocausal().is_cocausal
    assert Equation(graph.to_diagram(), traced)
    assert Hypergraph.spiders(0, 0, x).to_diagram() == Trace(Id(x))


def test_make_right_monogamous():
    from discopy.frobenius import Hypergraph as H, Spider
    x = frobenius.Ty('x')
    graph = H.spiders(3, 2, x).make_right_monogamous()
    assert graph.is_right_monogamous
    assert graph.boxes == (Spider(1, 2, x), )
    assert graph.wires == ((0, 0, 0), (((0,), (1, 2)),), (1, 2))
    assert H.spiders(2, 0, x).make_right_monogamous().boxes == (Spider(1, 0, x), )


def test_CMap():
    x = Ty('x')
    assert CMap.merge(x, 3).boxes == (Merge(x, 3), )
    assert CMap.noise(x).boxes == (Noise(x), )
    assert Merge(x).to_map().to_diagram() == Merge(x)
