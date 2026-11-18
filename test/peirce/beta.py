from itertools import product

from pytest import raises

from discopy.peirce.alpha import black, white
from discopy.peirce.beta import *
from discopy.utils import AxiomError

x = Ty('x')
R, S = Box('R', x, x), Box('S', x, x)


def test_coloured_wires():
    y = x.factory(Wire('x', dom=black))
    for colour, typ in ((white, x), (black, y)):
        for diagram in (Spider(1, 2, typ), Spider(2, 0, typ), Swap(typ, typ)):
            assert diagram.colour == colour
            assert diagram.negative().colour != colour
    with raises(AxiomError):
        Swap(x, y)
    assert (~R).arg.dom == y and (~R).dom == x


def test_dunders():
    assert R & S == Spider(1, 2, x) >> R @ S >> Spider(2, 1, x)
    assert R | S == ~(~R & ~S)
    assert R.relative_sum(S) == ~(~R >> ~S)
    assert R.par(S) == ~(~R @ ~S)


def test_model():
    for r in product((0, 1), repeat=4):
        M = Model(ob={'x': 2}, ar={'R': r})
        assert M(~~R) == M(R) and M(R.negative()) == M(~R)
        assert (M(~R).array == ~M(R).array).all()
        assert M(Id(x)) <= M(R) or not M(R).array.diagonal().all()


def test_axioms_hold():
    model = Model(ob={'x': 3}, ar=dict.fromkeys("abcd", [1, 0, 1] * 3))
    assert len(Model.axioms) == 67
    for law in Model.axioms.values():
        assert law(model=model), law.name
