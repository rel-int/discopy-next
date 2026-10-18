from __future__ import annotations
from pytest import importorskip, raises

from discopy.python import Function

from discopy.markov import *
from discopy import *


def test_Discard_has_no_copies():
    x = Ty('x')
    assert Copy(x, 0) == Discard(x) == Discard(x, 0)
    with raises(ValueError):
        Discard(x, 2)


def test_spider_factory():
    with raises(ValueError):
        Diagram.spider_factory(2, 2, Ty('x'))


def test_Merge_dagger():
    assert Merge(Ty('x')).dagger() == Copy(Ty('x'))


def test_Discard():
    assert isinstance(Discard(Ty('x')), Discard)
    assert isinstance(Copy(Ty('x'), n=0), Discard)
    x = Ty('x')
    assert str(Copy(x, 0)) == "Discard(x)" == str(Diagram.discard(x))
    assert eval(str(Discard(x))) == Copy(x, 0)


def test_equations():
    x = Ty('x')
    copy, discard = Copy(x), Copy(x, 0)
    add, minus, zero = Box('+', x @ x, x), Box('-', x, x), Box('0', Ty(), x)

    add >> copy, copy @ copy >> x @ Swap(x, x) @ x >> add @ add
    add >> discard, discard @ discard
    zero >> discard, Diagram.id(Ty())
    copy >> minus @ x >> add, discard >> zero, copy >> x @ minus >> add

    Diagram.id(x)
    x @ zero >> x @ copy >> add @ x >> discard @ x
    x @ zero @ zero >> discard @ discard @ x
    discard >> zero


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
        cod=Function)

    assert F(network)(42, 43) == max(0, sum([42 * 0, 43 * 1, 42 * 2, 43 * 3, -1]))


def test_Permutation():
    x, y, z = map(Ty, "xyz")
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    assert isinstance(perm, Box) and perm.cod == z @ x @ y
    assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    assert isinstance(perm.inside[0], Layer)
    assert Box('f', x, y).inside[0].boxes_or_types == (Box('f', x, y), )
    assert type(perm.inside[0].boxes_and_types[1]) is Permutation
    assert Permutation(x @ y, [1, 0]) == Swap(x, y)
    assert issubclass(Swap, Permutation)
    assert Equation(perm, perm.to_swaps())


def test_copy_and_merge_serialisation():
    from discopy.utils import dumps, loads
    x = Ty('x')
    for box in (Copy(x), Copy(x, 3), Merge(x), Merge(x, 0), Discard(x)):
        assert loads(dumps(box)) == box
    assert loads(dumps(Copy(x) >> Merge(x))) == Copy(x) >> Merge(x)
    assert repr(Merge(x, 3)) == "markov.Merge(monoidal.Ty(cat.Ob('x')), 3)"


def test_copy_and_merge_factories():
    from discopy.utils import factory

    @factory
    class Recipe(Diagram):
        """ A subclass with its own copies and merges. """

    class RecipeCopy(Copy, Recipe):
        """ The copy of an ingredient. """

    class RecipeMerge(Merge, Recipe):
        """ The merge of two ingredients. """

    Recipe.copy_factory, Recipe.merge_factory = RecipeCopy, RecipeMerge
    x = Ty('x')
    assert isinstance(RecipeCopy(x).dagger(), RecipeMerge)
    assert isinstance(RecipeMerge(x).dagger(), RecipeCopy)
    assert isinstance(Recipe.merge(x).boxes[0], RecipeMerge)


def test_Square_commutes():
    from hypothesis import given, settings, strategies as st

    from discopy import closed, feedback

    categories = st.sampled_from([Diagram, closed.Diagram, feedback.Diagram])

    @settings(max_examples=50, deadline=None)
    @given(st.data(), categories, st.booleans())
    def check(data, category, written):
        f, h, g = data.draw(Square[category].strategy(written=written))
        dom, cod = f.dom[:-len(h.dom)], f.cod[:-len(h.dom)]
        assert h.is_strict
        assert category.equation_factory(f >> cod @ h, dom @ h >> g)
    check()


def test_Square_keeps_flags():
    from discopy import closed

    for category in (Diagram, closed.Diagram):
        A, B, M, W = map(category.ob, "ABMW")
        box = category.box_factory
        sample = box('sample', A @ M, B @ W, is_deterministic=False)
        condition = box('condition', W, W, is_affine=False)
        f = sample >> B @ condition >> B @ box('write', W, M)
        for n in (2, 3):
            h, g = category.copy(M, n), Square[category].double(M, n)(f)
            assert set(g.boxes) >= {sample, condition}
            assert category.equation_factory(f >> B @ h, A @ h >> g)
            assert category.trace_uniformity_written(
                square=Square[category]((f, h, g)))


def test_Square_refuses_hypergraph_categories():
    from discopy import frobenius

    with raises(NotImplementedError):
        Square[frobenius.Diagram].strategy()


def test_Box_flags():
    import pickle
    from discopy.utils import dumps, loads
    p, x = Ty('p'), Ty('x')
    sample = Box('sample', p, x, is_deterministic=False)
    condition = Box('condition', x, Ty(), is_affine=False)
    assert Box('sample', p, x) != sample
    assert hash(Box('sample', p, x)) != hash(sample)
    assert repr(Box('f', p, x)) == "markov.Box('f', monoidal.Ty(cat.Ob('p')), "\
        "monoidal.Ty(cat.Ob('x')))"
    for box in (sample, condition, sample.dagger(), sample @ condition):
        assert eval(repr(box)) == box
        assert loads(dumps(box)) == box == pickle.loads(pickle.dumps(box))
    assert not sample.dagger().is_deterministic
    assert not condition.dagger().dagger().is_affine
    assert Copy(x).is_cartesian and Swap(x, x).is_cartesian
    assert Id(x).is_cartesian and Merge(x).is_deterministic
    assert not Merge(x).is_affine and not Merge(x).is_cartesian
    assert [(Merge(x, n).is_deterministic, Merge(x, n).is_affine)
            for n in range(3)] == [(False, False), (True, True), (True, False)]
    f = Box('f', x, x)
    assert not f.dagger().is_cartesian and f.dagger().dagger() == f
    assert not (Copy(p) >> sample @ p).is_deterministic
    assert (Copy(p) >> sample @ p).is_affine
    assert not (sample >> condition).is_cartesian
    assert loads(dumps(Box('f', p, x))).is_cartesian


def test_Box_flags_cooperate():
    import numpy as np
    from discopy import closed, frobenius, grammar, rigid, tensor
    from discopy.grammar.pregroup import Ty as Grammatical, Word
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
        'Word': Word, 'grammar': grammar, 'rigid': rigid}) == sample
    constant = closed.Constant('c', closed.Ty('x'))
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
