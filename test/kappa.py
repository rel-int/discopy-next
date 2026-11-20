from hypothesis import find
from pytest import raises

from discopy import closed, symmetric
from discopy.kappa import Constant, Kappa, TermBase, Variable, let
from discopy.monoidal import Box, Diagram, Functor, Ty
from discopy.utils import dumps, loads


x, y, z = map(Ty, "xyz")
f, g, h = Box('f', x, y), Box('g', x, z), Box('h', y @ z, x)
a, b, c, d = (Variable(name, x) for name in "abcd")


def test_one_output():
    effect, state = Box('e', x, Ty()), Box('s', Ty(), x)
    for diagram in (f @ g, effect, Diagram.id(Ty()), Diagram.id(x @ y)):
        with raises(ValueError):
            diagram.to_term()
    with raises(ValueError):
        Variable('a', Ty())
    assert Diagram.id(x).to_term() == x.kappa(lambda x0: x0)
    assert f.to_term() == x.kappa(lambda x0: Constant(f)(x0))
    assert state.to_term() == Constant(state)


def test_pack():
    X, Y, Z = map(closed.Ty, "XYZ")
    left, right = closed.Box('f', X, Y), closed.Box('g', X, Z)
    with raises(ValueError):
        TermBase.from_diagram(left @ right)
    diagram = left @ right >> closed.Pack(Y * Z)
    term = TermBase.from_diagram(diagram)
    assert term.eval() == diagram and term.category is closed.Diagram
    assert str(term.normal_form())\
        == "(X @ X).kappa(lambda x0, x1: Pack((Y * Z))(f(x0), g(x1)))"


def test_eval():
    k, e = Constant(Box('k', x @ y @ x, z)), Constant(Box('e', y, Ty()))
    v = Variable('v', y)
    assert Kappa((b, ), k(a, Constant(f)(b), c)).eval()\
        == x @ f @ x >> k.box
    term = let(e(v), lambda: k(a, Constant(f)(b), c), offset=2)
    assert term.freevars == [a, b, v, c]
    assert term.eval() == x @ x @ e.box @ x >> x @ f @ x >> k.box
    assert term.dom == x @ x @ y @ x and term.cod == z


def test_eval_partial_application():
    k = Constant(Box('k', x @ y @ z @ x, x))
    v, w = Variable('v', y), Variable('w', z)
    term = Kappa((v, w), k(a, v, w, c))(Constant(f)(b))
    assert term.freevars == [a, b, c] and term.offset == 2
    assert term.stack == z and term.dom == x @ x @ z @ x
    assert term.eval() == x @ f @ z @ x >> k.box
    assert term.normal_form() == z.kappa(lambda w: k(a, Constant(f)(b), w, c))


def test_functor():
    doubling = Functor(
        {x: x @ x, y: y, z: Ty()},
        lambda box: Box(
            box.name.upper(), doubling(box.dom), doubling(box.cod)))
    k, e = Box('k', x @ y, y), Box('e', x, Ty())
    diagram = x @ f @ x >> x @ y @ e >> k
    term = diagram.to_term()
    assert term.eval(doubling) == doubling(diagram)
    assert term.normal_form().eval(doubling).dom == x ** 6


def test_symmetric():
    X = symmetric.Ty('x')
    merge = symmetric.Box('m', X @ X, X)
    diagram = symmetric.Diagram.swap(X, X) >> merge
    term = diagram.to_term()
    assert term.eval() == diagram
    assert isinstance(term.eval(), symmetric.Diagram)


def test_normal_form_interchange():
    state, k = Box('s', Ty(), x), Box('k', x @ x, y)
    diagram = f @ g @ state >> h @ x >> k
    normal_form = diagram.to_term().normal_form()
    assert str(normal_form)\
        == "(x @ x).kappa(lambda x0, x1: k(h(f(x0), g(x1)), s))"
    steps = [step for left in (False, True)
             for step in diagram.normalize(left=left)]
    assert len({str(step) for step in steps}) > 2
    for step in steps:
        assert step.to_term() != diagram.to_term()
        assert step.to_term().normal_form() == normal_form


def test_normal_form_keeps_several_outputs():
    copy, k = Box('copy', x, x @ x), Box('k', x @ x @ x @ x, x)
    first = copy @ x >> x @ x @ copy >> k
    second = x @ copy >> copy @ x @ x >> k
    assert first.to_hypergraph() == second.to_hypergraph()
    for diagram in (first, second):
        term = diagram.to_term()
        assert term.normal_form() == term
    assert first.to_term() != second.to_term()


def test_substitute_shifts_offset():
    k, e, m = (Constant(Box(name, dom, cod)) for name, dom, cod in [
        ('k', x @ x, x), ('e', y, Ty()), ('m', x @ x, x)])
    v, s = Variable('v', y), Constant(Box('s', Ty(), x))
    body = lambda bound: let(e(v), lambda: m(bound, d), offset=1)
    longer = let(k(a, c), body).normal_form()
    assert longer == let(e(v), lambda: m(k(a, c), d), offset=2)
    assert longer.freevars == [a, c, v, d]
    shorter = let(s, body).normal_form()
    assert shorter == let(e(v), lambda: m(s, d), offset=0)
    assert shorter.freevars == [v, d]


def test_substitute_avoids_capture():
    m, n = Constant(Box('m', x @ x, x)), Constant(Box('n', x, x))
    term = let(n(b), lambda a: x.kappa(lambda b: m(a, b)))
    normal_form = term.normal_form()
    assert normal_form.freevars == term.freevars == [b]
    assert str(normal_form) == "x.kappa(lambda b_: m(n(b), b_))"
    assert normal_form.eval() == term.eval() == n.box @ x >> m.box
    assert x.kappa(lambda b: m(a, b)).substitute({c: d})\
        == x.kappa(lambda b: m(a, b))


def test_alpha():
    m = Constant(Box('m', x @ x, x))
    first = (x @ x).kappa(lambda a, b: m(a, b))
    second = (x @ x).kappa(lambda b, a: m(b, a))
    assert first != second and first.alpha_eq(second)
    assert not m(a, b).alpha_eq(m(b, a)) and not m(a, b).alpha_eq(a)
    assert not first.alpha_eq(m)
    assert not let(m(a, b), lambda c: c).alpha_eq(m(a, b))
    assert not Kappa((), m(a, b), 0).alpha_eq(Kappa((), m(a, b), 2))
    assert hash(first) != hash(second) and first != "first"


def test_application_errors():
    m = Constant(Box('m', x @ y, x))
    with raises(ValueError):
        m(Variable('v', y))
    with raises(ValueError):
        m(m)
    with raises(ValueError):
        m(a, Variable('v', y))(b)
    with raises(ValueError):
        Constant(Box('n', x @ x, x))(a, a)


def test_kappa_errors():
    m = Constant(Box('m', x @ x @ x, x))
    for variables, body, offset in [
            ((a, c), m(a, b, c), None), ((b, a), m(a, b), None),
            ((a, ), m(a, b, c), 1), ((d, ), m(a, b, c), None),
            ((), m(a, b, c), None), ((), m(a, b, c), 4),
            ((a, ), m(a, b), None), ((), m(a, b), 1)]:
        with raises(ValueError):
            Kappa(variables, body, offset)
    assert Kappa((b, ), m(a, b)).stack == x @ x
    assert Kappa((), m(a, b), 2).stack == x
    with raises(ValueError):
        (x @ x).kappa(lambda a: a)


def test_laws():
    laws = (
        TermBase.eta, TermBase.reading, TermBase.normal_form_soundness,
        TermBase.normal_form_idempotence, TermBase.repr_transparency,
        TermBase.str_transparency, TermBase.serialisation)
    shapes = [
        [1, 0, 0, 2, 2, 1, 2], [1, 2, 1, 2, 2, 0, 2], [2, 2, 2, 0, 2, 1, 1],
        [0, 1, 1, 2, 1, 2, 0], [1, 1, 2, 0, 1, 1, 0, 2, 0], [2, 1, 1, 1, 0]]
    for shape in shapes:
        term = TermBase.generate(y, shape, [x, z], "ab")
        assert all(law(term) for law in laws)
        renamed = TermBase.generate(y, shape, [x, z], "c")
        assert term.alpha_eq(renamed) and term.eval() == renamed.eval()
        assert loads(dumps(term)) == term
    term = find(TermBase.strategy(), lambda term: term.stack and "let" in str(
        term))
    assert all(law(term) for law in laws)
