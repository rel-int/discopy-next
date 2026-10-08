from pytest import raises
from random import choice, seed

from discopy import *
from discopy.feedback import *


def test_invalid_inputs():
    with raises(NotImplementedError):
        Ty('x').delay(-1)
    with raises(ValueError):
        HeadOb(Wire('x').delay())
    with raises(ValueError):
        TailOb(Wire('x').delay())


def test_Diagram_repr():
    x = Ty('x')
    plus = Box('plus', x @ x, x)
    zero = Box('zero', Ty(), x.head)
    zero, one = Box('zero', Ty(), x.head), Box('one', Ty(), x.head)
    fib =  ((
            Copy(x) >> one @ Diagram.wait(x) @ x
            >> FollowedBy(x) @ x >> plus).delay()
        >> zero @ x.delay() >> FollowedBy(x) >> Copy(x)).feedback()
    assert eval(str(fib)) == fib
    assert eval(repr(fib)) == fib


def test_functor_python_stream():
    x = Ty('x')
    zero, wait = Box('zero', Ty(), x), Diagram.wait(x)
    F = Functor(
        ob_map={x: int},
        ar_map={zero: lambda: 0},
        cod=stream.Stream[python.Function])
    assert F(wait @ zero).unroll(2).now(1, 2, 3) == (0, ) + (1, 0) + (2, 0) + (3, )


def test_walk():
    seed(420)
    X, fby = Ty('X'), FollowedBy(Ty('X'))
    zero, rand, plus = Box('0', Ty(), X), Box('rand', Ty(), X), Box('+', X @ X, X)

    @Diagram.feedback
    @Diagram.from_callable(X.d, X @ X)
    def walk(x):
        y = fby(zero.head(), plus.d(rand.d(), x))
        return (y, y)

    F = Functor(
        ob_map={X: int},
        ar_map={zero: lambda: 0,
            rand: lambda: choice([-1, +1]),
            plus: lambda x, y: x + y},
        cod=stream.Stream[python.Function])

    assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 2, 1, 0, -1, 0, 1)
    assert F(walk).unroll(9).now()[:10] == (0, -1, -2, -1, 0, 1, 0, 1, 2, 1)
    assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 0, 1, 0, -1, 0, -1)

def test_fibonacci():
    X = Ty('X')
    fby, wait = FollowedBy(X), Swap(X, X.d).feedback()
    zero, one = Box('0', Ty(), X), Box('1', Ty(), X)
    copy, plus = Copy(X), Box('+', X @ X, X)


    @Diagram.feedback
    @Diagram.from_callable(X.d, X @ X)
    def fib(x):
        y = fby(zero.head(), plus.d(fby.d(one.head.d(), wait.d(x)), x))
        return (y, y)

    fib_ = (
        copy.d >> one.head.d @ wait.d @ X.d
            >> fby.d @ X.d
            >> plus.d
            >> zero.head @ X.d
            >> fby >> copy).feedback()
    assert Equation(fib.arg, fib_.arg)

    F = Functor(
        ob_map={X: (int, )},
        ar_map={zero: lambda: 0,
            one: lambda: 1,
            plus: lambda x, y: x + y},
        cod=stream.Stream[python.Function])

    assert F(fib).unroll(9).now()[:10] == (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)


def test_Permutation_delay():
    x, y, z = map(Ty, "xyz")
    perm = Permutation(x @ y @ z, [2, 0, 1])
    assert perm.delay() == Permutation((x @ y @ z).delay(), [2, 0, 1])
    assert perm.delay(2) == perm.delay().delay()
    assert (perm >> Swap(z, x) @ y).delay()\
        == perm.delay() >> Swap(z, x).delay() @ y.delay()


def test_discard_is_a_feedback_diagram():
    x = Ty('x')
    discard = Diagram.copy(x, n=0)
    assert isinstance(discard, Diagram) and isinstance(discard, Discard)
    assert (discard >> Diagram.id(Ty())).boxes == [discard]


def test_Wire_to_tree():
    x = Wire('x')
    assert x.to_tree() == {'factory': 'feedback.Wire', 'name': 'x'}
    assert Wire('x', 2, is_constant=False).to_tree() == {
        'factory': 'feedback.Wire', 'name': 'x', 'time_step': 2,
        'is_constant': False}


def test_Feedback():
    x = Ty('x')
    f = Box('f', x, x)
    assert f.delay(2).time_step == 2
    with raises(ValueError):
        (f >> f).time_step
    assert Copy(x).delay() == Copy(x.delay())
    with raises(AxiomError):
        Box('g', x.delay() @ x, x @ x).feedback(dom=x, cod=x, mem=x)
    with raises(AxiomError):
        Box('g', x @ x.delay(), x @ x.d).feedback(dom=x, cod=x, mem=x)
    h = Box('h', x.delay() @ x, x @ x)
    loop = h.feedback(dom=x, cod=x, mem=x, left=True)
    assert h.feedback(x, x, x, True) == loop
    k = Box('k', x @ x.delay(), x @ x)
    assert k.feedback(x, x, x, False) == k.feedback(x, x, x)
    assert Functor({x: x}, {h: h})(loop) == loop
    with raises(AxiomError):
        loop.dagger()


def test_strategy():
    from hypothesis import find
    from discopy import pattern
    assert find(Wire.strategy(pattern.Obj(size=1)),
                lambda wire: True).time_step == 0
    assert len(find(Ty.strategy(pattern.Obj(size=2)), bool)) == 2
    assert find(Ty.strategy(pattern.Hom()), bool).dom == monoidal.transparent
