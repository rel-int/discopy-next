from pytest import importorskip, raises

from discopy import cat, grammar, monoidal, rigid
from discopy.grammar.cfg import Tree, Rule, Word, Id, Ty
from discopy.utils import AxiomError, dumps, loads

x = Ty('x')
f, w = Rule(x @ x, x, name='f'), Word('w', x)


def test_eq_and_hash():
    tree = f(f(w, w), Id(x))
    assert tree.cod == x and tree == f(f(w, w), Id(x)) and tree != 5
    assert {tree: 1, f: 2, w: 3}[f(f(w, w), Id(x))] == 1
    assert Tree(w) == w == Tree(w) and hash(Tree(w)) == hash(w)
    spider = Word('w', x, draw_as_spider=True, color='red')
    assert spider.to_diagram().draw_as_spider
    assert f(spider, spider).to_diagram().boxes[0].color == 'red'


def test_repr():
    namespace = dict(cat=cat, monoidal=monoidal, grammar=grammar)
    for tree in (f, f[::-1], w, Id(x), f(w, w), f(f(w, w), w)):
        assert eval(repr(tree), dict(namespace)) == tree


def test_dagger():
    assert (f[::-1].dom, f[::-1].cod) == (x, x @ x)
    assert (w[::-1].dom, w[::-1].cod) == (x, Ty())
    assert f[::-1][::-1] == f != f[::-1] and w[::-1][::-1] == w
    assert Id(x)[::-1] == Id(x) and str(f[::-1]) == "f[::-1]"
    assert f[::-1].to_diagram() == f.to_diagram()[::-1]


def test_from_nltk():
    nltk = importorskip("nltk")
    tree = Tree.from_nltk(nltk.Tree.fromstring("(S (NP I) (VP saw (NP him)))"))
    vp, np = Ty('VP'), Ty('NP')
    assert tree.branches[1] == Rule(vp @ np, vp, name='VP')(
        Word('saw', vp), Word('him', np))
    assert str(Tree.from_nltk(nltk.Tree.fromstring("(NP the big dog)")))\
        == "NP(the, big, dog)"


def test_Tree():
    x, y = Ty('x'), Ty('y')
    f, g = Rule(x @ x, x, name='f'), Rule(x @ y, x, name='g')
    a, b = Word('a', x), Word('b', y)
    tree = f(g(a, b), a)
    assert str(tree) == "f(g(a, b), a)" and tree.cod == x
    assert (tree.root, tree.branches) == (f, (g(a, b), a))
    with raises(AxiomError, match=r"f does not compose with \(a\): x @ x"):
        Tree(f, a)
    with raises(AxiomError, match=r"with \(a, b\): x @ x != x @ y"):
        f(a, b)
    with raises(AxiomError, match=r"rigid.Ty\(.*\) != monoidal.Ty\("):
        Rule(rigid.Ty('x') @ rigid.Ty('x'), x, name='r')(a, a)


def test_Tree_dom():
    a = Word('a', x)
    assert f(f, w).dom == x @ x and f(w, w).dom == Ty()
    assert f(f(f, f), f).dom == x ** 6
    assert f(f(f, f), f)(*6 * [a]) == f(f(f(a, a), f(a, a)), f(a, a))


def test_Tree_to_diagram():
    x, y = Ty('x'), Ty('y')
    f, g = Rule(x @ x, x, name='f'), Rule(x @ y, x, name='g')
    a, b = Word('a', x), Word('b', y)
    box = monoidal.Box
    leaf_a, leaf_b = box('a', Ty(), x), box('b', Ty(), y)
    assert f(g(a, b), a).to_diagram()\
        == (leaf_a @ leaf_b >> box('g', x @ y, x)) @ leaf_a\
        >> box('f', x @ x, x)


def test_Algebra():
    from discopy.grammar.cfg import Algebra, Operad
    y = Ty('y')
    g = Rule(y, x, name='g')
    identity = Algebra(lambda ob: ob, lambda ar: ar, cod=Operad())
    assert identity(f(g, g)) == f(g, g)
    with raises(TypeError):
        identity(42)


def test_Algebra_no_tree():
    from discopy.grammar.cfg import Algebra
    with raises(TypeError, match="Object of type Operad is not JSON"):
        dumps(Algebra({Ty('s'): Ty('s')}, {}))


def test_to_tree():
    x, y = Ty('x'), Ty('y')
    f, g = Rule(x @ x, x, name='f'), Rule(x @ y, x, name='g')
    a, b = Word('a', x), Word('b', y, data=3)
    for tree in (f(g(a, b), a), Tree(a), Tree(g[::-1], a), f, g[::-1], b,
                 Id(x), f(g, Id(x)), Tree(Id(x), a)):
        assert loads(dumps(tree)) == tree
        assert type(loads(dumps(tree))) is type(tree)
    raw = '{"factory": "grammar.cfg.Word", "name": "a", '\
        '"dom": {"factory": "monoidal.Ty", "inside": []}, '\
        '"cod": {"factory": "monoidal.Ty", "inside": '\
        '[{"factory": "monoidal.Wire", "name": "x"}]}}'
    assert loads(raw) == a and dumps(a) == raw
