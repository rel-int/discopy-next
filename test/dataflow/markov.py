from random import Random

from discopy import feedback, python, stream
from discopy.dataflow import symmetric
from discopy.dataflow.markov import *


def test_a_dataflow_diagram_is_a_markov_feedback_diagram():
    x = Ty('x')
    f = Box('f', x @ x.delay(), x @ x)
    for box in (f.feedback(), Box('g', x @ x, x @ x).trace(), Copy(x),
                Swap(x, x), f.head, FollowedBy(x)):
        assert isinstance(box, feedback.markov.Diagram)
        assert isinstance(box, symmetric.Diagram)
        assert isinstance(box, Diagram)
    assert type(f.feedback()).__module__ == "discopy.dataflow.markov"
    assert Diagram.feedback_factory is Feedback
    assert not hasattr(feedback.markov.Diagram, "followed_by")
    assert not hasattr(feedback.markov.Ty('x'), "head")


def test_stream_Diagram_repr():
    x = Ty('x')
    plus = Box('plus', x @ x, x)
    zero, one = Box('zero', Ty(), x.head), Box('one', Ty(), x.head)
    fib = ((
        Copy(x) >> one @ Diagram.wait(x) @ x
        >> FollowedBy(x) @ x >> plus).delay()
        >> zero @ x.delay() >> FollowedBy(x) >> Copy(x)).feedback()
    assert eval(str(fib)) == fib
    assert eval(repr(fib), {'dataflow': __import__('discopy.dataflow').dataflow,
                            'feedback': feedback}) == fib


def test_walk():
    rng = Random(420)
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
            rand: lambda: rng.choice([-1, +1]),
            plus: lambda x, y: x + y},
        cod=stream.Stream[python.Function])

    assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 2, 1, 0, -1, 0, 1)
    assert F(walk).unroll(9).now()[:10] == (0, -1, -2, -1, 0, 1, 0, 1, 2, 1)
    assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 0, 1, 0, -1, 0, -1)


def test_fibonacci():
    from discopy.markov import Equation
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

