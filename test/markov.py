from __future__ import annotations
import warnings
from inspect import Parameter, signature

from pytest import importorskip, raises, warns

from discopy.markov import *
from discopy import *
from discopy.utils import AxiomError


def test_Discard_has_no_copies():
    x = Ty('x')
    assert Copy(x, 0) == Discard(x) == Discard(x, 0)
    with raises(ValueError):
        Discard(x, 2)
    for box in (Copy, Discard):
        with raises(ValueError):
            box(x, -1)


def test_spider_factory():
    x = Ty('x')
    assert Diagram.spider_factory(1, 3, x) == Copy(x, 3)
    for n_legs_in, n_legs_out, phase in [(2, 2, None), (2, 1, None), (1, 2, .5)]:
        with raises(ValueError, match="only spiders of markov.Diagram"):
            Diagram.spider_factory(n_legs_in, n_legs_out, x, phase)


def test_Copy_dagger():
    """ A Markov category is not a dagger category, see #94 on next. """
    x = Ty('x')
    for box in (Copy(x), Copy(x, 3), Discard(x)):
        with raises(AxiomError, match="not a dagger category"):
            box.dagger()


def test_deprecated_merge():
    x = Ty('x')
    with warns(DeprecationWarning, match="use discopy.comarkov.Merge"):
        assert markov.Merge is comarkov.Merge
    with warns(DeprecationWarning, match="use discopy.comarkov.Diagram.merge"):
        assert Diagram.merge(x) == comarkov.Merge(x)
    with raises(AttributeError):
        markov.Codiagonal
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frobenius.Diagram.merge(frobenius.Ty('x'))
        neural.network.Network.merge(neural.network.Dims(2))


def test_Discard():
    assert isinstance(Discard(Ty('x')), Discard)
    assert isinstance(Copy(Ty('x'), n=0), Discard)
    x = Ty('x')
    assert str(Copy(x, 0)) == "Discard(x)" == str(Diagram.discard(x))
    assert eval(str(Discard(x))) == Copy(x, 0)


def test_equations():
    """ The laws of an abelian group in a Markov category, in the integers. """
    x = Ty('x')
    copy, discard = Copy(x), Copy(x, 0)
    add, minus, zero = Box('+', x @ x, x), Box('-', x, x), Box('0', Ty(), x)
    F = Functor({x: int}, {
        add: lambda a, b: a + b, minus: lambda a: -a, zero: lambda: 0},
        cod=python.Function)

    for sides in [
            (add >> copy, copy @ copy >> x @ Swap(x, x) @ x >> add @ add),
            (add >> discard, discard @ discard),
            (zero >> discard, Diagram.id(Ty())),
            (copy >> minus @ x >> add, discard >> zero,
             copy >> x @ minus >> add),
            (x @ zero >> x @ copy >> add @ x >> discard @ x,
             x @ zero @ zero >> discard @ discard @ x,
             discard >> zero)]:
        for xs in [(3, 5), (-2, 7)]:
            xs = xs[:len(sides[0].dom)]
            assert len({F(side)(*xs) for side in sides}) == 1


def test_neural_network():
    x = Ty('x')
    add = lambda n: Box('$+$', x ** n, x)
    ReLU = Box('$\\sigma$', x, x)
    weights = [Box('w{}'.format(i), x, x) for i in range(4)]
    bias = Box('b', Ty(), x)

    network = Diagram.copy(x @ x, 2)\
    >> Diagram.tensor(*weights) @ bias >> add(5) >> ReLU

    F = Functor(ob_map={x: int}, ar_map={
            add(5): lambda *xs: sum(xs),
            ReLU: lambda x: max(0, x),
            bias: lambda: -1, **{
                weight: lambda x, w=w: x * w
                for weight, w in zip(weights, range(4))}},
        cod=python.Function)

    assert F(network)(42, 43) == max(0, sum([42 * 0, 43 * 1, 42 * 2, 43 * 3, -1]))


def test_Permutation():
    x, y, z = map(Ty, "xyz")
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    assert isinstance(perm, Box) and perm.cod == z @ x @ y
    assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    assert isinstance(perm.inside[0], Layer)
    assert Box('f', x, y).inside[0].boxes_or_types == (Box('f', x, y), )
    assert type(perm.inside[0].boxes[0]) is Permutation
    assert Permutation(x @ y, [1, 0]) == Swap(x, y)
    assert issubclass(Swap, Permutation)
    assert Equation(perm, perm.to_swaps())


def test_copy_serialisation():
    from discopy.utils import dumps, loads
    x = Ty('x')
    for box in (Copy(x), Copy(x, 3), Discard(x)):
        assert loads(dumps(box)) == box
    assert loads(dumps(Copy(x) >> Discard(x) @ x)) == Copy(x) >> Discard(x) @ x
    assert repr(Copy(x, 3)) == "markov.Copy(monoidal.Ty(monoidal.Wire('x')), 3)"
    tree = dumps(comarkov.Merge(x, 3)).replace("comarkov.Merge", "markov.Merge")
    with warns(DeprecationWarning):
        assert loads(tree) == comarkov.Merge(x, 3)


def test_copy_factory():
    from discopy.utils import factory

    @factory
    class Recipe(Diagram):
        """ A subclass with its own copies. """

    class RecipeCopy(Copy, Recipe):
        """ The copy of an ingredient. """

    class RecipeDiscard(Discard, RecipeCopy):
        """ The discard of an ingredient. """

    Recipe.copy_factory, Recipe.discard_factory = RecipeCopy, RecipeDiscard
    x = Ty('x')
    assert isinstance(Recipe.copy(x).boxes[0], RecipeCopy)
    assert isinstance(RecipeCopy(x, 0), RecipeDiscard)
    with raises(AxiomError, match="Recipe is not a dagger category"):
        RecipeCopy(x).dagger()


def test_Copy_takes_index():
    from numpy import int64
    from discopy.abc import Nat
    x = Ty('x')
    assert Copy(x, int64(3)) == Copy(x, Nat(3)) == Copy(x, 3)
    assert Function(x, int64([0, 0, 0])) == Copy(x, 3)
    zero = type("Zero", (), {"__index__": lambda _: 0})()
    for box in (Copy(x, zero), Function(x, int64([]))):
        assert isinstance(box, Discard) and box == Discard(x)


def test_Function_takes_index():
    from numpy import int64
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Function(x @ y, int64([1, 0, 0])) == Function(x @ y, [1, 0, 0])
    assert Function(x, int64([0])) == Function(x, [0])
    assert Permutation(x @ y @ z, int64([2, 0, 1]))\
        == Permutation(x @ y @ z, [2, 0, 1])


def test_Copy_pickle():
    from pickle import dumps, loads
    x = Ty('x')
    for box in (Copy(x), Copy(x, 3), Discard(x)):
        assert loads(dumps(box)) == box


def test_hypergraph():
    x = Ty('x')
    f, g, h = Box('f', Ty(), x), Box('g', Ty(), x), Box('h', x, x)
    graph = (f @ g >> x @ Copy(x)).to_hypergraph()
    assert graph.offsets == (0, 1)
    assert graph.to_diagram() == Diagram(
        (Layer(f, g), Layer(x, Copy(x))), Ty(), x @ x @ x)
    diagram = f @ g >> Copy(x) @ x >> h @ x @ x
    assert diagram.to_hypergraph().offsets == (0, 1, None)
    assert diagram.to_hypergraph().to_diagram() == Diagram(
        (Layer(f, g), Layer(Copy(x), x), Layer(h, x, x)), Ty(), x @ x @ x)
    diagram = f @ g >> Discard(x) @ x
    assert diagram.to_hypergraph().to_diagram() == diagram
    graph = (f >> Copy(x) >> h @ h).to_hypergraph()\
        >> (x @ x @ g).to_hypergraph()
    assert graph.offsets == (0, None, None, 2)
    assert graph.to_diagram() == Diagram(
        (Layer(f), Layer(Copy(x)), Layer(h, h, g)), Ty(), x @ x @ x)


def test_from_callable():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x @ x, y)

    @Diagram.from_callable(x @ x, y)
    def diagram(a, b):
        f(b, a)
        return f(a, b)

    assert diagram == Diagram((
        Layer(Function(x @ x, [0, 1, 0, 1])),
        Layer(x, f, x), Layer(x, Discard(y), x), Layer(f)), x @ x, y)


def test_functor_python_stream():
    from discopy.feedback.markov import Box, Diagram, Functor, Ty
    x = Ty('x')
    zero, wait = Box('zero', Ty(), x), Diagram.wait(x)
    F = Functor(
        ob_map={x: int},
        ar_map={zero: lambda: 0},
        cod=stream.Stream[python.Function])
    assert F(wait @ zero).unroll(2).now(1, 2, 3) == (0, ) + (1, 0) + (2, 0) + (3, )


def test_Permutation_delay():
    from discopy.feedback.markov import Permutation, Swap, Ty
    x, y, z = map(Ty, "xyz")
    perm = Permutation(x @ y @ z, [2, 0, 1])
    assert perm.delay() == Permutation((x @ y @ z).delay(), [2, 0, 1])
    assert perm.delay(2) == perm.delay().delay()
    assert (perm >> Swap(z, x) @ y).delay()\
        == perm.delay() >> Swap(z, x).delay() @ y.delay()


def test_Discard_delay():
    from discopy.feedback.markov import Diagram, Discard, Ty
    x = Ty('x')
    discard = Diagram.copy(x, n=0)
    assert isinstance(discard, Diagram) and isinstance(discard, Discard)
    assert (discard >> Diagram.id(Ty())).boxes == [discard]
    assert discard.delay() == Discard(x.delay())


def test_delayed_Copy_pickle():
    from discopy.feedback.markov import Copy, Discard, Ty
    from pickle import dumps, loads
    x = Ty('x')
    for box in (Copy(x), Discard(x), Copy(x).delay(), Discard(x).delay()):
        assert loads(dumps(box)) == box


def test_Function():
    x, y = Ty('x'), Ty('y')
    f = Function(x @ y, [1, 0, 0])
    assert f.cod == y @ x @ x and f.size == 1 and isinstance(f, Box)
    assert Function(x, [0, 0, 0]) == Copy(x, 3)
    assert isinstance(Swap(x, y), Function) and isinstance(Copy(x), Function)
    assert Diagram.copy(x @ y, 0) == Discard(x) @ Discard(y)
    assert Equation(f, Swap(x, y) >> y @ Copy(x))
    with raises(ValueError):
        Function(x @ y, [2])
    with raises(ValueError):
        Function(x, [1])
    for box in (f, Copy(x), Swap(x, y)):
        assert eval(repr(box)) == box


def test_Function_serialisation():
    from pickle import dumps as pickle_dumps, loads as pickle_loads
    from discopy.utils import dumps, loads
    x, y = Ty('x'), Ty('y')
    f = Function(x @ y, [1, 0, 0])
    assert loads(dumps(f)) == f == pickle_loads(pickle_dumps(f))
    assert loads(dumps(f >> y @ Discard(x) @ x)) == f >> y @ Discard(x) @ x


def test_Function_Functor():
    x, y = Ty('x'), Ty('y')
    f = Function(x @ y, [1, 0, 0])
    F = Functor({x: int, y: str}, {}, cod=python.Function)
    assert F(f)(1, "a") == ("a", 1, 1)
    assert (f >> Box('g', y @ x @ x, x)).to_hypergraph().to_diagram()\
        == f >> Box('g', y @ x @ x, x)


def test_Functor_sends_a_Function_to_one_Function():
    x, y = Ty('x'), Ty('y')
    f = Function(x @ y, [1, 0, 0])
    assert Functor.id(Diagram)(f) == f
    G = Functor({x: x @ y, y: x}, {})
    assert G(f) == Function(x @ y @ x, [2, 0, 1, 0, 1])


def test_Layer():
    x, y = Ty('x'), Ty('y')
    g = Box('g', x, y)
    assert Layer(Copy(x), Copy(y)).boxes_or_types == (
        Function(x @ y, [0, 0, 1, 1]), )
    assert Layer(x, g, Copy(x)).is_plumbing
    diagram = Copy(x) @ y >> x @ Swap(x, y) >> Discard(x) @ y @ x
    assert diagram.foliation() == Function(x @ y, [1, 0])\
        == Swap(x, y) == diagram.foliation().inside[0].boxes_or_types[0]
    assert (Copy(x) >> Discard(x) @ x).foliation() == Id(x)


def test_Box_flags():
    import pickle
    from discopy.utils import dumps, loads
    p, x = Ty('p'), Ty('x')
    sample = Box('sample', p, x, is_deterministic=False)
    condition = Box('condition', x, Ty(), is_affine=False)
    assert Box('sample', p, x) != sample
    assert hash(sample) == hash(Box('sample', p, x, is_deterministic=False))
    assert repr(Box('f', p, x)) == "markov.Box('f', "\
        "monoidal.Ty(monoidal.Wire('p')), monoidal.Ty(monoidal.Wire('x')))"
    for box in (sample, condition, sample.dagger(), sample @ condition):
        assert eval(repr(box)) == box
        assert loads(dumps(box)) == box == pickle.loads(pickle.dumps(box))
    assert not sample.dagger().is_deterministic
    assert not condition.dagger().dagger().is_affine
    assert Copy(x).is_cartesian and Swap(x, x).is_cartesian
    assert Id(x).is_cartesian
    f = Box('f', x, x)
    assert not f.dagger().is_cartesian and f.dagger().dagger() == f
    assert not (Copy(p) >> sample @ p).is_deterministic
    assert (Copy(p) >> sample @ p).is_affine
    assert not (sample >> condition).is_cartesian
    assert loads(dumps(Box('f', p, x))).is_cartesian


def test_Box_flags_cooperate():
    import numpy as np
    from discopy import closed, frobenius, lambek, rigid, tensor
    from discopy.lambek.pregroup import Ty as Grammatical, Word
    from discopy.utils import dumps, loads
    x, d = frobenius.Ty('x'), tensor.Dim(2)
    f, flagged = frobenius.Box('f', x, x), frobenius.Box(
        'f', x, x, is_affine=False)
    assert f.r[::-1][::-1] == f.r and f.r.dagger().z == 1
    assert flagged.r.l == flagged and not flagged.r.dagger().is_affine
    assert not f.r.is_cartesian and f.r.l.is_cartesian
    for m, k in [(m, k) for m in range(3) for k in range(3)]:
        spider = frobenius.Spider(m, k, x)
        assert (spider.is_deterministic, spider.is_affine) == (m > 0, m == 1)
        assert spider.r.l == spider and loads(dumps(spider)) == spider
    assert not frobenius.Spider(1, 1, x, 0.5).is_deterministic
    with raises(TypeError):
        frobenius.Spider(1, 2, x, is_affine=False)
    cup, cap = frobenius.Cup(x, x), frobenius.Cap(x, x)
    assert cup.is_deterministic and not cup.is_affine
    assert not cap.is_deterministic and not cap.is_affine
    n, word = Grammatical('n'), Word('w', Grammatical('n'))
    assert (word.dagger().dom, word.dagger().cod) == (n, Grammatical())
    sample = Word('w', n, is_deterministic=False)
    assert sample.r.l == sample and sample.r.flags() == sample.flags()
    assert eval(repr(sample), {
        'Word': Word, 'lambek': lambek, 'rigid': rigid}) == sample
    constant = closed.markov.Constant('c', closed.markov.Ty('x'))
    assert loads(dumps(constant)) == constant
    t = tensor.Box('t', d, d, data=[[1, 2j], [3, 4]])
    assert np.allclose(t.r.dagger().eval().array, t.r.eval().dagger().array)
    assert t != tensor.Box('t', d, d, data=t.data, is_deterministic=False)


def test_Box_flags_subs():
    importorskip("sympy")
    from sympy.abc import phi
    from discopy import tensor
    x, d = Ty('x'), tensor.Dim(2)
    sample = Box('sample', x, x, data=phi, is_deterministic=False)
    condition = tensor.Box('c', d, d, data=[phi, 0, 0, 1], is_affine=False)
    assert sample.subs(phi, 1) == Box(
        'sample', x, x, data=1, is_deterministic=False)
    samples = (sample.dagger().lambdify(phi)(1),
               (x @ sample >> sample @ x).subs(phi, 1))
    conditions = (condition.subs(phi, 1), condition.r.lambdify(phi)(1))
    assert not any(diagram.is_deterministic for diagram in samples)
    assert not any(diagram.is_affine for diagram in conditions)
    assert condition.subs(phi, 1).flags() == {'is_affine': False}


def test_Bubble_flags():
    from discopy.egraph import Markov
    from discopy.utils import dumps, loads
    x = Ty('x')
    noisy = Box('sample', x, x, is_deterministic=False).bubble(
        is_deterministic=False, is_affine=False)
    assert noisy[::-1].affine is False
    plain = Box('sample', x, x, is_deterministic=False).bubble(is_affine=False)
    assert loads(dumps(plain)) == plain == eval(repr(plain))
    assert eval(repr(plain[::-1])).is_dagger
    assert plain != plain.arg.bubble()
    minus, zero = Box('-', x @ x, x), Box('0', Ty(), x)
    program = Diagram.copy(x) >> noisy @ noisy >> minus
    graph = Markov.from_diagram(program)
    *_, last = graph, *graph.saturate(
        (Diagram.copy(x) >> minus, Diagram.discard(x) >> zero))
    assert last.extract().to_diagram() == program.foliation()


def test_Bubble_flags_subs():
    importorskip("sympy")
    from sympy.abc import phi
    x = Ty('x')
    sample = Box('sample', x, x, data=phi, is_deterministic=False)
    noisy = sample.bubble(is_deterministic=False, is_affine=False)
    assert not noisy.subs(phi, 1).is_deterministic
    assert not noisy.lambdify(phi)(1).affine


def test_Box_signature():
    sig = signature(Box)
    assert list(sig.parameters) == [
        "name", "dom", "cod", "data", "is_dagger", "args", "method",
        "is_deterministic", "is_affine"]
    x = Ty('x')
    box = Box('f', x, x, **{
        name: param.default for name, param in sig.parameters.items()
        if param.default is not Parameter.empty})
    assert box == Box('f', x, x)


def test_structural_boxes():
    from discopy.testable import assert_structural_boxes
    assert_structural_boxes(Diagram)


def test_term_copies_a_variable_used_twice():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y @ y, x)
    diagram = Diagram.copy(x) >> f @ f >> g
    term = diagram.to_term()
    assert str(term) == (
        "x(lambda x0: let(f(x0), lambda x1: "
        "let(f(x0), lambda x2: g(x1, x2))))")
    assert Equation(term.eval(), diagram)
    assert str(term.normal_form()) == "x(lambda x0: g(f(x0), f(x0)))"
    assert Equation(term.normal_form().eval(), diagram)
    a = Variable('a', x)
    twice = Constant(Box('m', x @ x, y))(a, a)
    assert twice.freevars == [a] and twice.count(a) == 2
    assert twice.eval() == Copy(x) >> Box('m', x @ x, y)
    with raises(ValueError, match="disjoint free variables"):
        symmetric.Constant(symmetric.Box('m', x @ x, y))(
            symmetric.Variable('a', x), symmetric.Variable('a', x))


def test_term_keeps_a_statement_copied_or_discarded():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y @ y, x)
    shared = f >> Diagram.copy(y) >> g
    term = shared.to_term().normal_form()
    assert str(term) == "x(lambda x0: let(f(x0), lambda x1: g(x1, x1)))"
    assert Equation(term.eval(), shared)
    assert not Equation(shared, Diagram.copy(x) >> f @ f >> g)
    unused = f @ x >> Diagram.discard(y) @ x
    term = unused.to_term().normal_form()
    assert str(term) == (
        "(x @ x)(lambda x0, x1: let(f(x0), lambda x2: x1, offset=0))")
    assert Equation(term.eval(), unused)
    assert not Equation(unused, Diagram.discard(x) @ x)
    assert eval(str(term), TermBase.str_environment(term)) == term
    a, b = Variable('a', x), Variable('b', x)
    redex = let(Constant(f)(a), lambda v: b, offset=1)
    assert redex.normal_form() == redex and redex.freevars == [b, a]
    assert redex.eval() == x @ f >> x @ Discard(y)


def test_term_kappa_discards_what_its_body_does_not_use():
    x, y = Ty('x'), Ty('y')
    f, a, b = Constant(Box('f', x, y)), Variable('a', x), Variable('b', y)
    term = Kappa((b, a), f(a))
    assert term == (y @ x)(lambda b, a: f(a)) and type(term) is Kappa
    assert term.stack == y @ x and term.eval() == Discard(y) @ f.box
    assert Kappa((b, ), f(a), 1).eval() == x @ Discard(y) >> f.box
    assert Kappa((b, ), f(a), 0).eval() == Discard(y) @ x >> f.box
    assert str(Kappa((b, ), f(a), 1)) == "y(lambda b: f(a), offset=1)"
    with raises(ValueError, match="Expected an offset"):
        Kappa((b, ), f(a))
    with raises(ValueError, match="offset from 0 to 1"):
        Kappa((b, ), f(a), 2)
    with raises(ValueError, match="distinct variables to bind in f.a."):
        Kappa((a, a), f(a))
    low = symmetric.Constant(symmetric.Box('f', x, y))
    with raises(ValueError, match="distinct free variables"):
        (y @ x)(lambda b, a: low(a))
    assert type((y @ x)(lambda b, a: a)) is Kappa
    assert type(x(lambda a: a)) is monoidal.Kappa


def test_term_substitute_under_a_kappa_of_no_variable():
    x, y = Ty('x'), Ty('y')
    e, k = Constant(Box('e', x, Ty())), Constant(Box('k', x @ x, y))
    a, b = Variable('a', x), Variable('b', x)
    term = (x @ x)(lambda a, b: let(e(a), lambda: k(b, a), offset=0))
    assert term.freevars == [] and term.eval() == Function(
        x @ x, [0, 1, 0]) >> e.box @ x @ x >> k.box
    assert TermBase.reading(term) and TermBase.normal_form_soundness(term)
    assert term.eval().to_term().alpha_eq(term)
    body = let(e(a), lambda: k(b, a), offset=0)
    assert body.context == ([a, b], Ty(), [])
    assert body.func.context == ([], Ty(), [b, a])


def test_term_functor_into_tensors():
    import numpy as np
    from discopy.tensor import Dim, Tensor
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y @ y, x)
    arrays = {f: np.array([[1, 2], [3, 4]]), g: np.arange(8).reshape(2, 2, 2)}
    F = Functor({x: Dim(2), y: Dim(2)}, arrays, cod=Tensor[int])
    shared = (f >> Diagram.copy(y) >> g).to_term()
    copied = (Diagram.copy(x) >> f @ f >> g).to_term()
    assert (shared.eval(F).array == np.einsum(
        "ab,bbc->ac", arrays[f], arrays[g])).all()
    assert (copied.eval(F).array == np.einsum(
        "ab,ac,bcd->ad", arrays[f], arrays[f], arrays[g])).all()
    assert (shared.eval(F).array != copied.eval(F).array).any()
    unused = (f @ x >> Diagram.discard(y) @ x).to_term().normal_form()
    assert (unused.eval(F).array == np.einsum(
        "a,cd->acd", arrays[f].sum(axis=1), np.eye(2, dtype=int))).all()
    dropped = (Diagram.discard(x) @ x).to_term()
    assert (unused.eval(F).array != dropped.eval(F).array).any()


def test_term_laws():
    x, y = Ty('x'), Ty('y')
    laws = (
        TermBase.eta, TermBase.reading, TermBase.normal_form_soundness,
        TermBase.normal_form_idempotence, TermBase.repr_transparency,
        TermBase.str_transparency, TermBase.serialisation)
    shapes = [
        [1, 2, 1, 1, 0], [2, 1, 1, 2, 0, 1, 1, 0, 1], [0, 1, 2, 2, 1, 2, 1],
        [1, 0, 1, 1, 2, 2, 0, 1, 2], [2, 0, 2, 1, 1, 3, 1, 2, 0, 1, 0, 1]]
    wired = 0
    for shape in shapes:
        term = TermBase.generate(y, shape, [x, y], "ab")
        assert all(law(term) for law in laws)
        wired += any(isinstance(box, Function) for box in term.eval().boxes)
        renamed = TermBase.generate(y, shape, [x, y], "c")
        assert term.alpha_eq(renamed)
        assert Equation(term.eval(), renamed.eval())
    assert wired
    v, w = Variable('v', x), Variable('w', x)
    assert str(Sampler(TermBase, iter([0, 0, 1]), [x], "a").state(
        y, (v, w))) == "c0(v, w, v)"
    assert str(Sampler(TermBase, iter([0, 1, 2]), [x], "a").state(
        y, (v, w))) == "c0(v)"
