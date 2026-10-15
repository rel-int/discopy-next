from pytest import raises

from discopy.grammar.cfg import *
from discopy.utils import AxiomError


def test_Tree():
    x, y = Ty('x'), Ty('y')
    f, g = Rule(x @ x, x, name='f'), Rule(y, x, name='g')
    with raises(AxiomError):
        Tree(f, g)  # The branches must give the domain of the root.
    tree = f(g, g)
    assert repr(tree).startswith("grammar.cfg.Tree(f, *(")
    assert repr(Id(x)) == "Id(x)" and Tree.id(x) == Id(x)
    assert f == tree.root and f != tree


def test_Algebra():
    x, y = Ty('x'), Ty('y')
    f, g = Rule(x @ x, x, name='f'), Rule(y, x, name='g')
    identity = Algebra(lambda ob: ob, lambda ar: ar, cod=Operad())
    assert identity(f(g, g)) == f(g, g)
    with raises(TypeError):
        identity(42)
