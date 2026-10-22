from pytest import raises

from discopy.frobenius import *


def test_Functor_call():
    x = Ty('x')
    cup, cap = Cup(x, x.r), Cap(x.r, x)
    box1, box2 = Box('box', x, x), Box('box', x @ x, x @ x)
    spider = Spider(0, 2, x)
    F = Functor(lambda x: x @ x, {box1: box2})
    assert F(cup) == Id(x) @ cup @ Id(x.r) >> cup
    assert F(cap) == Id(x.r) @ cap @ Id(x) << cap
    assert F(box1) == box2
    assert F(box1.l) == box2.l and F(box1.r) == box2.r
    assert F(spider) == spider @ spider >> Id(x) @ Swap(x, x) @ Id(x)
    with raises(TypeError):
        F(F)


def test_Functor_phased_spider():
    x = Ty('x')
    phased, box = Spider(1, 1, x, .5), Box('box', x, x)
    F = Functor({x: x}, {phased: box})
    assert F(phased) == box and F(Spider(1, 1, x)) == Spider(1, 1, x)
    assert Functor.id(Diagram)(phased) == phased
    assert not Equation(phased, Spider(1, 1, x, .25))
    spider = Spider(1, 2, x, .5)
    with raises(KeyError):
        Functor(lambda x: x, {})(spider)
    assert spider.to_hypergraph().boxes == (spider, )


def test_Box_hash():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    assert f == f @ Id()
    assert hash(f) == hash(f @ Id())
    assert hash(f) == hash(Id() @ f)
    assert f @ Id() in {f}
    assert {f: 42}[f @ Id()] == 42
    s = Spider(1, 2, x)
    assert hash(s) == hash(s @ Id())


def test_spider_adjoint():
    n = Ty('n')
    one = Box('one', Ty(), n)
    two = Box('two', Ty(), n)
    diagram = one @ two >> Spider(2, 1, n)

    assert diagram.r == diagram.l == Spider(1, 2, n) >> two.r @ n >> one.r


def test_spider_factory():
    a, b, c = map(Ty, 'abc')
    ts = [a, a @ b, a @ b @ c]
    for i in range(5):
        for j in range(5):
            for t in ts:
                s = Diagram.spiders(i, j, t)
                for k, ob in enumerate(t):
                    assert all(map(ob.__eq__, s.dom[k::len(t)]))
                    assert all(map(ob.__eq__, s.cod[k::len(t)]))


def test_spider_decomposition():
    n = Ty('n')

    assert Spider(0, 0, n).unfuse() == Spider(0, 1, n) >> Spider(1, 0, n)
    assert Spider(1, 0, n).unfuse() == Spider(1, 0, n)
    assert Spider(1, 1, n).unfuse() == Id(n)
    assert Spider(2, 1, n).unfuse() == Spider(2, 1, n)

    # 5 is the smallest number including both an even and odd decomposition
    assert Spider(5, 1, n).unfuse() == (Spider(2, 1, n) @ Spider(2, 1, n)
                                           @ Id(n) >> Spider(2, 1, n) @ Id(n)
                                           >> Spider(2, 1, n))


def test_Sum():
    f = Box('f', 'x', 'x')
    assert Sum([f]) == f
    assert isinstance(f + f, Sum) and (f + f).terms == (f, f)
    image = Functor(lambda x: x, {f: f + f})(f)
    assert isinstance(image, Sum) and image.terms == (f, f)


def test_Spider_to_tree():
    from discopy.utils import dumps, loads
    x = Ty('x')
    for spider in (Spider(1, 2, x), Spider(2, 0, x, 0.5), Spider(0, 3, x).l,
                   Spider(1, 2, x, is_dagger=True)):
        assert loads(dumps(spider)) == spider
        assert loads(dumps(spider)).phase == spider.phase
        assert loads(dumps(spider)).is_dagger == spider.is_dagger


def test_Spider_repr():
    from discopy import frobenius
    x = Ty('x')
    assert repr(Spider(1, 2, x, 0.5))\
        == "frobenius.Spider(1, 2, frobenius.Ty(frobenius.Wire('x')), 0.5)"
    for spider in (Spider(1, 2, x, 0.5), Spider(2, 0, x, is_dagger=True),
                   Spider(0, 1, x, 0.25, is_dagger=True)):
        assert eval(repr(spider)) == spider


def test_Wire_adjoints():
    from discopy.monoidal import Colour
    a, b = Colour('a'), Colour('b')
    x = Wire('x', dom=a, cod=b)
    for adjoint in (x.l, x.r):
        assert (adjoint.name, adjoint.dom, adjoint.cod) == ('x', b, a)
        assert adjoint.l == adjoint.r == x
    assert Ty(x).l == Ty(x).r == Ty(x.l) and Ty(x).l.dom == b
    assert Wire('y').l == Wire('y').r == Wire('y')


def test_feedback_is_the_trace():
    x = Ty('x')
    f = Box('f', x @ x, x @ x)
    for traced in (f.feedback(), f.feedback(mem=x), f.trace(mem=x)):
        assert isinstance(traced, Diagram) and traced == f.trace()
    assert Equation(f.feedback(2, left=True), f.trace(2, left=True))
