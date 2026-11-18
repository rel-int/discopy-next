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
        assert (M(Id(x)) <= M(R)) == M(R).array.diagonal().all()


def test_axioms_hold():
    from random import Random
    random, models = Random(0), []
    for n in (1, 2, 3):
        for _ in range(3):
            models.append(Model(ob={'x': n}, ar={
                name: [random.random() < .5 for _ in range(n * n)]
                for name in "abcd"}))
    assert len(Model.axioms) == 67
    for law in Model.axioms.values():
        for model in models:
            assert law(model=model), (law.name, model)


def test_rotated_cut():
    model = Model(ob={'x': 2}, ar={'R': [[1, 1], [0, 0]]})
    assert model((~R).r) == model(~R).r and model((~R).l) == model(~R).l


def test_boundaries():
    from discopy.peirce.alpha import white
    R2, S2 = Box('R2', x @ x, x @ x), Box('S2', x @ x, x @ x)
    model = Model(ob={'x': 2}, ar={'R2': [1, 0] * 8, 'S2': [1, 1, 0, 0] * 4})
    r, s = model(R2).array, model(S2).array
    assert (model(R2 & S2).array == (r & s)).all()
    assert (model(R2 | S2).array == (r | s)).all()
    p = Box('p', Ty.id(white), Ty.id(white))
    unit = Model(ar={'p': True})
    assert unit(p & p).array.all() and unit(p | ~p).array.all()
    assert Id().dom == Ty.id(white)


def test_flags():
    assert not R.is_deterministic and not R.is_affine
    assert not (~R).is_deterministic and not (~R).is_affine
    assert Box('f', x, x, is_deterministic=True).is_deterministic


def test_negative_keeps_flags():
    f = Box('f', x, x, is_deterministic=True, is_affine=True)
    assert f.negative().negative() == f and f.negative().is_deterministic


def test_cups_caps_traces():
    model = Model(ob={'x': 2}, ar={'R': [[1, 1], [0, 0]]})
    snake = x @ Diagram.caps(x, x) >> Diagram.cups(x, x) @ x
    assert model(snake) == model(Id(x))
    r = model(R).array.reshape(2, 2)
    assert (model((R @ x).trace()).array.reshape(2, 2) == r).all()
