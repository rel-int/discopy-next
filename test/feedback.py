from pytest import raises, warns

from discopy.feedback import *
from discopy.utils import AxiomError


def test_invalid_inputs():
    with raises(NotImplementedError):
        Ty('x').delay(-1)
    with raises(ValueError):
        HeadOb(Wire('x').delay())
    with raises(ValueError):
        TailOb(Wire('x').delay())


def test_trace_is_a_feedback_diagram():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x @ y, x @ y)
    for trace in (f.trace(), f.trace(left=True)):
        assert isinstance(trace, Trace) and isinstance(trace, Diagram)
        assert isinstance(trace >> trace, Diagram)
        assert trace.delay() == f.delay().trace(left=trace.left)
        assert trace.delay(2) == trace.delay().delay()
        assert trace.delay() >> trace.delay() == (trace >> trace).delay()


def test_feedback_with_a_memory_of_many_wires():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x @ (y @ z).delay(), x @ y @ z)
    for fb in (f.feedback(mem=y @ z), f.feedback(dom=x, cod=x, mem=y @ z)):
        assert fb == f.feedback(mem=z).feedback(mem=y)
        assert (fb.dom, fb.cod) == (x, x)
    assert f.feedback(mem=Ty()) == f


def test_delayed_Box_to_tree():
    from discopy.utils import dumps, loads
    x, y = Ty('x'), Ty('y')
    for box in (Box('f', x, y), Box('f', x, y, data=42).dagger()):
        for delayed in (box, box.delay(), box.delay(2)):
            assert loads(dumps(delayed)) == delayed
            assert loads(dumps(delayed)).time_step == delayed.time_step


def test_Feedback_to_tree():
    from discopy.utils import dumps, loads
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x @ (y @ z).delay(), x @ y @ z)
    for fb in (f.feedback(mem=z), f.feedback(mem=y @ z), Feedback(f, mem=y @ z)):
        assert loads(dumps(fb)) == fb
        assert loads(dumps(fb)).mem == fb.mem


def test_Head_Tail_FollowedBy_to_tree():
    from discopy.utils import dumps, loads
    x, y = Ty(Wire('x', is_constant=False)), Ty('y')
    f = Box('f', x, y)
    for term in (x.head, x.tail, x.head.delay(), f.head, f.tail, f.tail.delay(2),
                 FollowedBy(x), FollowedBy(x, is_dagger=True).delay()):
        assert loads(dumps(term)) == term


def test_trace_repr():
    from discopy import feedback
    trace = Box('f', 'x', 'x').trace()
    assert repr(trace) == "feedback.Trace(feedback.Box('f', "\
        "feedback.Ty(feedback.Wire('x')), feedback.Ty(feedback.Wire('x'))))"
    assert eval(repr(trace)) == trace
    assert repr(Box('f', 'x', 'x').trace(left=True)).endswith(", left=True)")


def test_trace_error():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_trace_out_of_range():
    from discopy import compact
    from discopy.matrix import Matrix
    from discopy.python import additive
    x = compact.Ty('x')
    f = compact.Box('f', x @ x, x @ x)
    g = additive.Function(lambda i, tag=0: (i, tag), (int, int), (int, int))
    for traced in (Box('f', 'x', 'x'), f, f.to_hypergraph(), f.to_drawing(),
                   Matrix[bool].swap(1, 1), g):
        for n in (3, -1):
            with raises(ValueError):
                traced.trace(n)


def test_trace_dagger():
    f = Box('f', 'x', 'x')
    assert f.trace().dagger() == f.dagger().trace()


def test_trace_vanishing():
    from discopy import compact, matrix, ribbon
    from discopy.python import additive, multiplicative

    x = compact.Ty('x')
    f = compact.Box('f', x @ x, x @ x)
    assert f.trace(0) == f
    assert f.to_hypergraph().trace(0) == f.to_hypergraph()
    assert f.to_map().trace(0) == f.to_map()
    assert f.to_drawing().trace(0) == f.to_drawing()

    y = ribbon.Ty('y')
    g = ribbon.Box('g', y @ y, y @ y)
    assert g.trace(0) == g

    assert matrix.Matrix[bool].swap(1, 1).trace(0)\
        == matrix.Matrix[bool].swap(1, 1)

    h = additive.Function(lambda i, tag=0: (i, tag), (int, int), (int, int))
    assert h.trace(0) == h

    k = multiplicative.Function(lambda i, j: (i, j), (int, int), (int, int))
    assert k.trace(0) == k


def test_Sum():
    f = Box('f', 'x', 'x')
    assert Sum([f]) == f
    assert isinstance(f + f, Sum) and (f + f).terms == (f, f)
    image = Functor(lambda x: x, {f: f + f})(f)
    assert isinstance(image, Sum) and image.terms == (f, f)


def test_trace_serialisation():
    from discopy.utils import dumps, loads
    from discopy import symmetric
    f = Box('f', 'x', 'x')
    for trace in (f.trace(), f.trace(left=True)):
        assert loads(dumps(trace)) == trace
    x = symmetric.Ty('x')
    swap = symmetric.Swap(x, x)
    assert loads(dumps(swap.trace(left=True))) == swap.trace(left=True)


def test_trace_foliation():
    x = Ty('x')
    f, g = Box('f', x @ x, x @ x), Box('g', x, x)
    beside, below = f.trace() @ g, f.trace() >> g
    assert beside.foliation().boxes == [f.trace(), g]
    assert len(beside.foliation()) == 1 and beside.depth() == 1
    assert below.foliation() == below and below.depth() == 2


def test_traced_is_deprecated():
    from discopy import feedback, traced
    with warns(DeprecationWarning):
        assert traced.Diagram is feedback.Diagram
    with warns(DeprecationWarning):
        assert traced.Trace is feedback.Trace


def test_feedback_forms():
    x, m = Ty('x'), Ty('m')
    f = Box('f', x @ m.delay(), x @ m)
    assert f.feedback() == f.feedback(1) == f.feedback(mem=m)\
        == f.feedback(cod=x) == f.feedback(dom=x, cod=x, mem=m)
    with raises(TypeError):
        f.feedback(1, mem=m)
    with raises(ValueError):
        f.feedback(3)


def test_trace_is_an_undelayed_feedback():
    x, m = Ty('x'), Ty('m')
    f, g = Box('f', x @ m, x @ m), Box('g', x @ m.delay(), x @ m)
    assert isinstance(f.feedback(), Trace) and f.feedback() == f.trace()
    assert isinstance(g.feedback(), Feedback)
    assert not isinstance(g.feedback(), Trace)
    assert str(f.trace(mem=m)) == "Trace(f)"
    assert str(g.feedback()) == "Feedback(g)"
    assert f.trace(dom=x, cod=x, mem=m) == f.trace()
    with raises(AxiomError):
        g.trace()
    with raises(AxiomError):
        Feedback(f)


def test_left_feedback():
    x, m = Ty('x'), Ty('m')
    f = Box('f', m.delay() @ x, m @ x)
    assert f.feedback(left=True) == Feedback(f, left=True)
    assert f.feedback(left=True).delay() == f.delay().feedback(left=True)
    assert str(f.feedback(left=True)) == "Feedback(f, left=True)"


def test_traced_category_feedback_is_its_trace():
    from discopy import compact
    x = compact.Ty('x')
    f = compact.Box('f', x @ x, x @ x).to_map()
    assert f.feedback() == f.feedback(mem=x) == f.feedback(1) == f.trace()
    assert f.feedback(mem=x, left=True) == f.trace(left=True)
    assert f.delay() is f
    with raises(TypeError):
        f.feedback(1, mem=x)


def test_every_trace_takes_the_keyword_form():
    from inspect import signature
    from discopy import ribbon
    from discopy.abc import keyword_form
    from discopy.matrix import Matrix
    from discopy.python import Function

    class Sub(Matrix[bool]):
        def trace(self, n=1, left=False):
            return super().trace(n, left)

    for trace in (Matrix.trace, Function.trace, Sub.trace,
                  ribbon.Diagram.trace, Diagram.trace):
        assert list(signature(trace).parameters)\
            == ["self", "n", "left", "dom", "cod", "mem"]
        assert keyword_form(trace) is trace
    f = Sub([[1, 1], [0, 0]], 2, 2)
    assert f.trace(mem=1) == f.trace() == Sub([[1]], 1, 1)
    assert f.trace(mem=1, left=True) == f.trace(left=True) == Sub([[0]], 1, 1)
    x = ribbon.Ty('x')
    g = ribbon.Box('g', x @ x, x @ x)
    assert g.trace(mem=x) == g.trace()
    assert g.trace(cod=x, left=True) == g.trace(left=True)
    with raises(TypeError):
        g.trace(1, mem=x)


def test_yanking():
    from discopy import balanced, compact, lens, para, pivotal, symmetric
    from discopy.abc import Nat, PivotalCategory
    from discopy.matrix import Matrix
    from discopy.axioms import AxiomFailure
    from discopy.drawing import Drawing
    from discopy.python import Function, Ty as PythonTy
    x, y = symmetric.Ty('x'), compact.Ty('y')
    assert symmetric.Diagram.yanking(x) and compact.Diagram.yanking(y)
    assert compact.CMap.yanking(y)
    yanked, identity = Function.yanking(PythonTy(int)).terms
    assert yanked(42) == identity(42) == 42
    assert para.Traced[Matrix[bool]].yanking(Nat(2))
    assert lens.Traced[Matrix[bool]].yanking(lens.Ty[Nat](Nat(1), Nat(2)))
    for planar in (Diagram, pivotal.Diagram, Drawing, PivotalCategory):
        assert planar.yanking() is NotImplemented
        assert "planar" in planar.yanking.__doc__
    with raises(AxiomFailure):
        balanced.Diagram.yanking(balanced.Ty('x'))


def test_types_without_wires_are_their_own_delay():
    from discopy import frobenius, monoidal, pivotal, rigid, symmetric
    from discopy.neural.network import Dims
    for x in (monoidal.Nat(2), rigid.Nat(2), Nat(2), symmetric.Nat(2),
              pivotal.Nat(2), frobenius.Nat(2), monoidal.Dim(2, 3),
              frobenius.Dim(2, 3), Dims(2, 3)):
        assert x.delay() == x.delay(2) == x.head == x.tail == x
    f = frobenius.Box('f', frobenius.Nat(1), frobenius.Nat(1))
    assert f.delay().dom == f.dom and f.delay().time_step == 1


def test_feedback_laws_on_a_traced_category():
    from discopy import compact
    from discopy.abc import FeedbackCategory
    C = compact.CMap
    x, y, m, n, w, z = map(compact.Ty, "xymnwz")
    box = lambda name, dom, cod: compact.Box(name, dom, cod).to_map()
    f = box('f', x @ m @ n, y @ m @ n)
    g, h, k, e = box('g', w, x), box('h', y, z), box(
        'k', m @ n, m @ n), box('e', w, z)
    for law, arguments in (
            ("vanishing", dict(f=f)),
            ("joining", dict(f=f, x=x, y=y, m=m, n=n)),
            ("strength", dict(e=e, f=f, x=x, y=y, m=m @ n)),
            ("tightening", dict(f=f, g=g, h=h, x=x, y=y, m=m @ n)),
            ("sliding", dict(f=f, k=k, x=x, y=y, m=m @ n))):
        axiom = getattr(FeedbackCategory, law).bind(C)
        assert axiom.modulo(C.to_hypergraph)(**arguments)


def test_split_memory_bounds():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, x)
    with raises(ValueError):
        f.split_memory(mem=x @ y)
    with raises(ValueError):
        f.feedback(mem=x @ y)


def test_delayed_dagger():
    from discopy import feedback
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y, data=42)
    assert f.delay().dagger() == f.dagger().delay()
    assert f.delay().dagger().time_step == 1
    assert f.delay().dagger().dagger() == f.delay()
    for box in (f.delay().dagger(), f.dagger().delay(2)):
        assert eval(repr(box)) == box


def test_traced_wildcard_import():
    import warnings
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from discopy.traced import *", namespace)
    assert namespace['Box'] is Box and namespace['Trace'] is Trace


def test_delayed_pivotal_diagrams():
    from discopy import pivotal
    x = pivotal.Ty('x')
    cup, cap = pivotal.Cup(x, x.r), pivotal.Cap(x.r, x)
    snake = x @ cap >> cup @ x
    assert snake.delay() == x.d @ pivotal.Cap(x.r.d, x.d)\
        >> pivotal.Cup(x.d, x.r.d) @ x.d
    assert cup.delay().dagger() == cup.dagger().delay()
    f = pivotal.Box('f', x, x @ x)
    for g in (f.delay().dagger(), f.delay().r, f.delay().l):
        assert g.time_step == 1
    assert f.delay().dagger() == f.dagger().delay()
    assert f.delay().r == f.r.delay()
    for box in (f.delay().dagger(), f.delay().r.dagger()):
        assert eval(repr(box), {'pivotal': pivotal}) == box


def test_time_stepped_wires_below_feedback():
    from discopy import markov, pivotal, frobenius
    x = markov.Ty('x')
    assert x.delay() != x and x.delay().inside[0].time_step == 1
    f = markov.Box('f', x @ x.delay(), x @ x)
    assert isinstance(f.feedback(), markov.Feedback)
    assert isinstance(markov.Box('g', x @ x, x @ x).trace(), markov.Trace)
    y = pivotal.Ty('y')
    assert y.delay().l == y.l.delay() and y.delay().l.l == y.delay()
    g = pivotal.Box('g', y @ y.delay(), y @ y)
    assert isinstance(g.feedback(), pivotal.Feedback)
    assert g.delay().time_step == 1
    assert frobenius.Ty('z').delay().r == frobenius.Ty('z').r.delay()


def test_FollowedBy_delay():
    x = Ty('x')
    for is_dagger in (False, True):
        fby = FollowedBy(x, is_dagger)
        assert fby.delay().delay() == fby.delay(2)\
            == FollowedBy(x, is_dagger, time_step=2)
        assert fby.delay(2).dom == fby.dom.delay(2)
        assert fby.delay(2).cod == fby.cod.delay(2)


def test_generated_boxes_at_every_level():
    import pickle
    from discopy import (
        balanced, closed, comarkov, compact, frobenius, markov, pivotal,
        ribbon, symmetric)
    from discopy.grammar import abstract
    from discopy.neural import network
    from discopy.utils import dumps, factory_name, loads
    for module in (balanced, symmetric, markov, comarkov, pivotal, ribbon,
                   compact, frobenius, closed, abstract, network):
        diagram = getattr(module, "Diagram", getattr(module, "Network", None))
        for name in ("Feedback", "Trace", "FollowedBy", "Head", "Tail"):
            cls = getattr(module, name)
            assert cls.__module__ == module.__name__ and cls.__doc__
            assert issubclass(cls, module.Box)
            assert issubclass(cls, globals()[name])
            assert factory_name(cls) == module.__name__.removeprefix(
                "discopy.") + "." + name
        assert diagram.feedback_factory is module.Feedback
        assert diagram.head_factory is module.Head
    for module in (balanced, symmetric, markov, comarkov):
        x = module.Ty('x')
        f = module.Box('f', x @ x.delay(), x @ x)
        for box in (f.feedback(), f.head, f.tail, module.Box(
                'g', x @ x, x @ x).trace(), module.FollowedBy(x)):
            assert type(box).__module__ == module.__name__
            assert eval(repr(box), {module.__name__.split('.')[-1]: module,
                                    'feedback': __import__(
                                        'discopy.feedback').feedback}) == box
            assert pickle.loads(pickle.dumps(box)) == box
            assert loads(dumps(box)) == box
    assert pivotal.Diagram.trace_factory.__func__\
        is pivotal.Diagram.__dict__['trace_factory'].__func__


def test_heads_of_pivotal_types():
    import pickle
    from discopy import compact, frobenius, pivotal
    from discopy.utils import dumps, loads
    for module in (pivotal, compact, frobenius):
        x = module.Ty('x')
        assert isinstance(x.head.inside[0], module.Wire)
        assert x.head.l == x.l.head and x.head.r == x.r.head
        assert x.head.delay().tail == x.head
        f = module.Box('f', x, x @ x)
        assert type(f.head).__module__ == module.Diagram.head_factory.__module__
        assert f.head.dom == x.head and f.head.cod == (x @ x).head
        for term in (x.head, x.head.delay(), f.head):
            assert pickle.loads(pickle.dumps(term)) == term
            assert loads(dumps(term)) == term


def test_structural_boxes_delay_through_their_arguments():
    import pickle
    from discopy import (
        balanced, comarkov, frobenius, markov, pivotal, symmetric)
    x, y = symmetric.Ty('x'), symmetric.Ty('y')
    for box, delayed in (
            (symmetric.Swap(x, y), symmetric.Swap(x.d, y.d)),
            (symmetric.Permutation(x @ y @ x, [2, 0, 1]),
             symmetric.Permutation((x @ y @ x).d, [2, 0, 1])),
            (markov.Copy(x, 3), markov.Copy(x.d, 3)),
            (markov.Discard(x), markov.Discard(x.d)),
            (comarkov.Merge(x), comarkov.Merge(x.d)),
            (balanced.Braid(x, y, is_dagger=True),
             balanced.Braid(x.d, y.d, is_dagger=True)),
            (balanced.Twist(x).dagger(), balanced.Twist(x.d).dagger())):
        assert box.delay() == delayed and box.delay().time_step == 0
        assert box.delay(2) == box.delay().delay()
        assert pickle.loads(pickle.dumps(box)) == box
    z = pivotal.Ty('z')
    for box in (pivotal.Cup(z, z.r), pivotal.Cap(z.r, z)):
        assert box.delay() == type(box)(z.d if box.left == z else z.r.d,
                                        z.r.d if box.left == z else z.d)
        assert pickle.loads(pickle.dumps(box)) == box
    w = frobenius.Ty('w')
    for is_dagger in (False, True):
        spider = frobenius.Spider(1, 2, w, .5, is_dagger=is_dagger)
        assert spider.delay() == frobenius.Spider(
            1, 2, w.d, .5, is_dagger=is_dagger)
        assert spider.delay().delay(-1) == spider
        assert pickle.loads(pickle.dumps(spider)) == spider
