from discopy.braided import *
from pytest import raises
from discopy.utils import AxiomError


x, y, z = map(Ty, "xyz")
a, b = Ty('a'), Ty('b')
f = Box('f', a, b)

def test_hexagon():
    assert Diagram.braid(x, y @ z) == Braid(x, y) @ z >> y @ Braid(x, z)
    assert Diagram.braid(x @ y, z) == x @ Braid(y, z) >> Braid(x, z) @ y


def test_simplify():
    assert (Diagram.braid(x, y @ z) >> Diagram.braid(x, y @ z)[::-1]).simplify()\
        == Diagram.id(x @ y @ z)\
        == (Diagram.braid(y @ z, x)[::-1] >> Diagram.braid(y @ z, x)).simplify()


def test_to_hypergraph():
    # A braid is not a swap: it stays a box of the hypergraph and of the map.
    from discopy import hypergraph, cmap
    for convert in (Diagram.to_hypergraph, Diagram.to_map):
        assert convert(Braid(x, y)).boxes == (Braid(x, y), )
        assert convert(Diagram.braid(x @ y, z)).boxes == (
            Braid(y, z), Braid(x, z))
    for representation in (hypergraph.Hypergraph, cmap.CMap):
        assert representation[Diagram].braid(x, y).boxes == (Braid(x, y), )
        with raises(AxiomError, match="not symmetric"):
            representation[Diagram].swap(x, y)


def test_functor_factory():
    assert Diagram.functor_factory is Functor
    F = Diagram.functor_factory({x: y, y: x}, {})
    assert F(Braid(x, y)) == Braid(y, x)
    assert F(Braid(x, y).dagger()) == Braid(y, x).dagger()


def test_Braid_to_tree():
    from discopy.utils import dumps, loads
    for braid in (Braid(x, y), Braid(x, y).dagger()):
        assert loads(dumps(braid)) == braid
        assert loads(dumps(braid)).is_dagger == braid.is_dagger
