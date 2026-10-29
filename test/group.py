import pickle
import random
from functools import reduce

from pytest import mark, raises

from discopy.group import (
    Action, Cyclic, Product, Symmetric, Wreath, kronecker)
from discopy.python.finset import Permutation

GROUPS = (
    Cyclic[1], Cyclic[4], Symmetric[0], Symmetric[1], Symmetric[4],
    Product[Cyclic[2], Symmetric[3]],
    Wreath[Cyclic[2], Symmetric[3]],
    Wreath[Symmetric[3], Cyclic[2]],
    Wreath[Product[Cyclic[2], Cyclic[2]], Cyclic[2]])

ACTIONS = tuple(G.action() for G in GROUPS) + (
    Wreath[Symmetric[3], Cyclic[2]].action(power=True),
    Wreath[Cyclic[2], Symmetric[3]].action(power=True),
    Product[Cyclic[2], Symmetric[3]].action(
        Cyclic[2].action().tensor(Cyclic[2].action()),
        Symmetric[3].action()))


@mark.parametrize("G", GROUPS)
def test_order(G):
    elements = list(G.elements())
    assert len(elements) == len(set(elements)) == G.order()


@mark.parametrize("G", GROUPS)
def test_word(G):
    for g in G.elements():
        assert reduce(G.then, (G.generators()[i] for i in g.word()), G()) == g


@mark.parametrize("G", GROUPS)
def test_group_laws(G):
    elements = list(G.elements())
    for g in elements:
        assert G() >> g == g == g >> G()
        assert g >> g.dagger() == G() == g.dagger() >> g
        for h in elements:
            assert (g >> h).dagger() == h.dagger() >> g.dagger()
            assert g @ h == g >> h


@mark.parametrize("action", ACTIONS, ids=repr)
def test_functoriality(action):
    elements = list(action.group.elements())
    assert action(action.group()) == Permutation.id(action.size)
    images = {g: action(g) for g in elements}
    for g in elements:
        for h in elements:
            assert images[g >> h] == images[g] >> images[h]


@mark.parametrize("G", GROUPS)
def test_sample(G):
    rng = random.Random(42)
    assert all(isinstance(G.sample(rng), G) for _ in range(10))


@mark.parametrize("G", GROUPS)
def test_pickle(G):
    g = list(G.elements())[-1]
    assert pickle.loads(pickle.dumps(g)) == g


def test_subscript():
    for G in (Cyclic, Symmetric):
        with raises(TypeError):
            G()
    for G in (Product, Wreath):
        with raises(TypeError):
            G()
    with raises(TypeError):
        Product[Cyclic[2], Cyclic[3]](Cyclic[3](), Cyclic[2]())
    with raises(ValueError):
        Wreath[Cyclic[2], Cyclic[3]]((Cyclic[2](), ))
    with raises(ValueError):
        Product[Cyclic[2], Cyclic[3]](Cyclic[2]())
    with raises(ValueError):
        Cyclic[0]()


def test_kronecker():
    p, q = Permutation((1, 2, 0)), Permutation((1, 0))
    assert kronecker(p, q) == (3, 2, 5, 4, 1, 0)
    assert kronecker(p, q) >> kronecker(q.dagger(), p.dagger())\
        != Permutation.id(6)
    assert kronecker(p, q) >> kronecker(p.dagger(), q.dagger())\
        == kronecker(p >> p.dagger(), q >> q.dagger()) == Permutation.id(6)


def test_action():
    action = Cyclic[3].action()
    assert eval(repr(action), Cyclic.environment()) == action
    with raises(ValueError):
        Action(Cyclic[3], 3, ())
    with raises(TypeError):
        action.tensor(Cyclic[2].action())
    with raises(TypeError):
        Product[Cyclic[3], Cyclic[2]].action(Cyclic[2].action())
    with raises(TypeError):
        Wreath[Cyclic[3], Cyclic[2]].action(Cyclic[2].action())


def test_orbits():
    """
    The orbits of a group on pairs of points are the orbitals: for the
    symmetric group, the diagonal and the rest, so that an equivariant
    linear map is a combination of the identity and the all-ones matrix.
    """
    action = Symmetric[4].action()
    diagonal, rest = action.tensor(action).orbits()
    assert diagonal == tuple(5 * i for i in range(4))
    assert len(rest) == 12
    trivial = Action(Symmetric[1], 3, ())
    assert trivial.orbits() == ((0, ), (1, ), (2, ))
    reversal = Action(Cyclic[2], 4, ((3, 2, 1, 0), ))
    assert reversal.orbits() == ((0, 3), (1, 2))
    assert Action(Cyclic[4], 4, ((1, 3, 0, 2), )).orbits() == ((0, 1, 2, 3), )


def test_sudoku():
    """
    The sudoku group, acting on cells and digits. Two lines, rows or
    columns, are equal, in the same band or not, and the transposition
    exchanges rows and columns, so a pair of cells is in one of six
    orbits, which span the linear maps on the cells commuting with it: the
    same cell, the rest of its row or column in its box, outside its box,
    the rest of its box, the rest of its band or stack, and the rest.
    """
    lines = Wreath[Symmetric[3], Symmetric[3]]
    cells = Wreath[lines, Cyclic[2]]
    assert cells.order() == 2 * 1296 ** 2
    grid = cells.action(power=True)
    assert grid.size == 81 and len(grid.orbits()) == 1
    assert len(grid.tensor(grid).orbits()) == 6
    rng = random.Random(0)
    g = cells.sample(rng)
    assert grid(g) >> grid(g.dagger()) == Permutation.id(81)
    sudoku = Product[cells, Symmetric[9]]
    assert sudoku.order() == 1218998108160
