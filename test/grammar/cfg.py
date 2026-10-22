from pytest import raises

from discopy import monoidal
from discopy.grammar.cfg import *
from discopy.utils import AxiomError


def test_Tree():
    x, y = Ty('x'), Ty('y')
    f, g = Rule(x @ x, x, name='f'), Rule(x @ y, x, name='g')
    a, b = Word('a', x), Word('b', y)
    tree = f(g(a, b), a)
    assert str(tree) == "f(g(a, b), a)" and tree.cod == x
    assert (tree.root, tree.branches) == (f, (g(a, b), a))
    with raises(AxiomError):
        Tree(f, a)


def test_Tree_to_diagram():
    x, y = Ty('x'), Ty('y')
    f, g = Rule(x @ x, x, name='f'), Rule(x @ y, x, name='g')
    a, b = Word('a', x), Word('b', y)
    box = monoidal.Box
    leaf_a, leaf_b = box('a', Ty(), x), box('b', Ty(), y)
    assert f(g(a, b), a).to_diagram()\
        == (leaf_a @ leaf_b >> box('g', x @ y, x)) @ leaf_a\
        >> box('f', x @ x, x)
