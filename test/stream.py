from pytest import raises

from discopy import *
from discopy.stream import *

def test_errors():
    T, S = Ty[python.Ty], Stream[python.Function]
    with raises(ValueError):
        S(lambda x: x, mem=T(int))
    dom = cod = mem = T(int)
    now = python.Function(lambda x, y: (x + y, x - y), (int, int), (int, int))
    with raises(AxiomError):
        S(now, dom, cod)
    with raises(AxiomError):
        S(now, dom, cod, mem.head)
    with raises(AxiomError):
        non_constant = T(dom, _later=lambda: dom)
        S(now, non_constant, cod, mem)
    mismatched_cod = python.Function(lambda x: x, (int, ), (int, ))
    with raises(AxiomError, match=r"str.*!= .*int"):
        S(mismatched_cod, T(int), T(str))


def test_python_stream():
    T, S = Ty[python.Ty], Stream[python.Function]
    x, y, m = int, bool, str
    dom = T(x) @ T(m).delay()
    cod = T(y) @ T(m)
    now = python.Function(lambda n: (bool(n % 2), str(n)), (x, ), (y, m))
    later = python.Function(
        lambda n, s: (bool(n % 2), s + " " + str(n)), (x, m), (y, m))
    s = S(now, dom, cod, _later=lambda: S(later))
    s.check_later()
    s.unroll().check_later()
    fb = s.feedback(dom=T(x), cod=T(y), mem=T(m))
    assert fb.unroll(2).now(1, 2, 3) == (True, False, True, '1 2 3')


def test_sequence_box_of_its_category():
    """ A stream builds the boxes of the category it is over. """
    from discopy import markov, tensor
    for category in (symmetric.Diagram, markov.Diagram):
        x = Ty.sequence(category.ob('x'))
        box = Stream[category].sequence("f", x, x).now
        assert type(box) is category.box_factory
    dim = Ty[tensor.Dim](tensor.Dim(2))  # A constant stream of dimensions.
    box = Stream[tensor.Diagram].sequence("f", dim, dim).now
    assert type(box) is tensor.Box is tensor.Diagram.box_factory


def test_sequence_of_a_name_is_one_wire():
    mem = Ty.sequence('mem')
    assert mem.now == Ty.sequence(symmetric.Ty('mem')).now
    assert mem.now @ mem.later.now == symmetric.Ty('mem0', 'mem1')


def test_feedback_errors():
    x = Ty.sequence(symmetric.Ty('x'))
    with raises(ValueError):  # A constant stream needs constant types.
        Stream(symmetric.Box('g', x.now, x.now), dom=x, cod=x)
    assert Stream.permutation([0, 1], [x, x]).now\
        == symmetric.Id(x.now @ x.now)


def test_feedback_forms():
    from pytest import raises
    x, y, m = (Ty.sequence(symmetric.Ty(name)) for name in "xym")
    f = Stream.sequence("f", x @ m.delay(), y @ m)
    unrolled = f.feedback(dom=x, cod=y, mem=m).unroll(2).now
    for fb in (f.feedback(), f.feedback(1), f.feedback(mem=m)):
        assert fb.unroll(2).now == unrolled
    with raises(TypeError):
        f.feedback(1, mem=m)
    with raises(NotImplementedError):
        f.feedback(left=True)
    with raises(NotImplementedError):
        f.feedback(cod=y)
    with raises(ValueError):
        f.feedback(3)


def test_feedback_of_no_memory():
    from pytest import raises
    from discopy.utils import AxiomError
    x, y = symmetric.Ty('x'), symmetric.Ty('y')
    constant = Stream(symmetric.Box('f', x, x))
    assert constant.feedback(0) == constant == constant.feedback(mem=Ty())
    with raises(AxiomError):
        constant.feedback(dom=Ty(y), cod=Ty(y), mem=Ty())
    with raises(AxiomError):
        Stream(symmetric.Box('g', x @ y, x @ y)).feedback(
            dom=Ty(y), cod=Ty(x), mem=Ty(y))
    f = Stream.sequence("f", *map(Ty.sequence, "xy"))
    assert f.feedback(0) is f and f.feedback(mem=Ty()) is f


def test_last_and_without():
    from pytest import raises
    X, Y = map(Ty.sequence, "xy")
    XY = X @ Y
    assert [XY.last(1).later.now, XY.without(Y).later.now] == [
        symmetric.Ty("y1"), symmetric.Ty("x1")]
    with raises(ValueError):
        XY.without(X)


def test_feedback_laws():
    from pytest import raises
    from discopy.abc import FeedbackCategory
    from discopy.testable import AxiomFailure
    x, y, m, n, w, z = (Ty.sequence(symmetric.Ty(name)) for name in "xymnwz")
    f = Stream.sequence('f', x @ (m @ n).delay(), y @ m @ n)
    g, h, k, e = (Stream.sequence(name, dom, cod) for name, (dom, cod) in zip(
        "ghke", [(w, x), (y, z), (m @ n, m @ n), (w, z)]))
    up_to = lambda stream: stream.unroll(2).now.to_hypergraph()
    assert FeedbackCategory.vanishing.bind(Stream).modulo(up_to)(f=f)
    for law, arguments in (
            ("joining", dict(f=f, x=x, y=y, m=m, n=n)),
            ("strength", dict(e=e, f=f, x=x, y=y, m=m @ n)),
            ("tightening", dict(f=f, g=g, h=h, x=x, y=y, m=m @ n))):
        assert getattr(FeedbackCategory, law).bind(Stream).modulo(up_to)(
            **arguments)
    sliding = dict(f=f, k=k, x=x, y=y)
    assert not FeedbackCategory.sliding.bind(Stream).modulo(up_to)(**sliding)
    with raises(AxiomFailure):
        Stream.sliding(**sliding)


def test_no_tree():
    from discopy.utils import dumps
    x = Ty(symmetric.Ty('x'))
    for stream in (Stream.sequence('f', x, x),
                   Stream(symmetric.Box('f', x.now, x.now))):
        with raises(TypeError, match="A stream has no tree"):
            dumps(stream)
    assert "serialisation" not in Stream.axioms


def test_singleton_equality_compares_the_callee():
    """ The two ``singleton``s put the callee itself in `_later` rather than
    a fresh lambda, so two built the same way hold the *same* object and
    compare equal, where every other non-constant stream closes over its
    arguments. Why `_later` is compared at all is the intensional equality
    of :class:`discopy.stream.Ty`, whose doctests state it. """
    x, y = symmetric.Ty('x'), symmetric.Ty('y')
    assert Ty.singleton(x) == Ty.singleton(x)
    assert Ty.singleton(x) != Ty.singleton(y)
    f = Stream.singleton(symmetric.Box('f', x, y))
    assert f == Stream.singleton(symmetric.Box('f', x, y))
    assert f.later == Stream.id() and f.now == symmetric.Box('f', x, y)


def test_no_hash():
    """ Streams and their types stay unhashable, with the ``__hash__ = None``
    a dataclass gives whatever it compares, see
    :class:`discopy.stream.Ty` for the equality it goes with. """
    x = Ty(symmetric.Ty('x'))
    for unhashable in (Stream.sequence('f', x, x), x):
        with raises(TypeError, match="unhashable"):
            hash(unhashable)


def test_delay_passes_the_initial_memory_through():
    x, y, m = (Ty.sequence(symmetric.Ty(name)) for name in "xym")
    f = Stream.sequence("f", x @ m.delay(), y @ m)
    for stream in (Stream.sequence("g", x, y, m), f.feedback().later):
        delayed = stream.delay()
        delayed.check_later()
        assert delayed.now == symmetric.Id(stream.mem.now)
        assert delayed.later is stream
        assert delayed.unroll().now.to_hypergraph()\
            == stream.now.to_hypergraph()
