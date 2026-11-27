# -*- coding: utf-8 -*-

import pickle

from pytest import raises

from discopy import monoidal, symmetric
from discopy.symmetric import *
from discopy.utils import AxiomError, dumps, loads


def test_Swap():
    x, y = Ty('x'), Ty('y')
    assert repr(Swap(x, y)) == "symmetric.Swap("\
        "monoidal.Ty(monoidal.Wire('x')), monoidal.Ty(monoidal.Wire('y')))"
    assert Swap(x, y).dagger() == Swap(y, x)
    with raises(ValueError):
        Swap(x ** 2, Ty())


def test_Swap_says_what_is_wrong():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(ValueError, match="of length 1, got length 2"):
        Swap(x @ y, z)
    with raises(ValueError, match=r"the swap \[1, 0\], got \[0, 1\]"):
        Swap(x @ y, [0, 1])


def test_Box_hash():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    assert f == f @ Id()
    assert hash(f) == hash(f @ Id())
    assert hash(f) == hash(Id() @ f)
    assert f @ Id() in {f}
    assert {f: 42}[f @ Id()] == 42


def test_symmetric_Equation():
    """
    ``symmetric.Equation`` compares diagrams up to hypergraph isomorphism
    (e.g. swaps cancel) while ``==`` stays syntactic,
    see https://github.com/discopy/discopy/issues/382
    """
    x = Ty('x')
    a, b = Swap(x, x) >> Swap(x, x), Id(x @ x)
    assert a != b
    assert Equation(a, b)
    assert not Equation(a, Swap(x, x))


def test_Box_hash_is_syntactic_and_stable():
    """
    Equality and hashing are always syntactic, so a box and the length-one
    diagram made of it are equal and hash equally, and a box can always be
    found in a dict, see https://github.com/discopy/discopy/issues/382
    """
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    assert f == f @ Id() and hash(f) == hash(f @ Id())
    assert f @ Id() in {f}
    assert {f: 42}[Box('f', x, y)] == 42


def test_Functor_keys_boxes_by_syntax():
    """
    Regression test for https://github.com/discopy/discopy/issues/382

    A ``Functor`` stores the image of each generator by hashing the boxes in
    its domain.  Now that equality and hashing are always syntactic, box
    lookups are stable and functor application never loses a box.
    """
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    F = Functor(ob_map={x: y, y: x}, ar_map={f: g, g: f})
    assert F(f) == g and F(g) == f
    assert F(f >> g) == g >> f


def test_Diagram_permutation(monkeypatch):
    x = Nat(1)
    monkeypatch.setattr(Diagram, "ob", Nat)
    assert Diagram.swap(x, x ** 2)\
        == Diagram.swap(x, x) @ Id(x) >> Id(x) @ Diagram.swap(x, x)\
        == Diagram.permutation([1, 2, 0])\
        == Diagram.permutation([2, 0, 1]).dagger()
    with raises(ValueError):
        Diagram.permutation([2, 0])
    with raises(ValueError):
        Diagram.permutation([2, 0, 1], x ** 2)


def test_bad_permute():
    with raises(ValueError):
        Id(Ty('n')).permute(1)
    with raises(ValueError):
        Id(Ty('n')).permute(0, 0)


def test_Permutation():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm = Permutation(x @ y @ z, [1, 2, 0])
    assert perm.dom == x @ y @ z and perm.cod == y @ z @ x
    assert list(perm.perm) == [1, 2, 0]
    assert perm.inside == (Layer(perm),)
    assert perm.boxes == [perm] and perm.size == 0
    assert perm.is_generator and perm.generator == perm
    assert perm.encode() == (perm.dom, [[(perm, 0)]])
    assert Diagram.decode(*perm.encode()) == perm
    identity = Permutation(x @ y @ z, [0, 1, 2])
    assert identity.is_identity and identity == Id(x @ y @ z)
    assert identity.inside == () and identity.boxes == []
    assert Permutation.id(x @ y @ z) == Id(x @ y @ z)
    assert Permutation.id(x @ y @ z).inside == ()
    assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    assert perm.dagger().dagger() == perm
    a = Permutation(x @ y @ z, [1, 2, 0])
    b = Permutation(a.cod, [2, 0, 1])
    c = Permutation(b.cod, [0, 2, 1])
    assert (a >> b) >> c == a >> (b >> c)
    assert Id(x @ y @ z) >> a == a == a >> Id(a.cod)
    assert (a >> b).dagger() == b.dagger() >> a.dagger()
    q = Permutation(z @ y, [1, 0])
    assert (perm @ q).dagger() == perm.dagger() @ q.dagger()
    assert (perm @ q).dom == perm.dom @ q.dom
    swap = Swap(x, y)
    assert Permutation(x @ y, [1, 0]) == swap
    assert isinstance(Swap(x, y), Permutation)
    assert swap.to_swaps() is swap
    with raises(ValueError):
        Permutation(x @ y @ z, [2, 0])
    with raises(ValueError):
        Permutation(x @ y @ z, [0, 0, 1])


def test_Layer():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x, y)
    swap, permutation = Swap(x, y), Permutation(x @ y, [1, 0])
    layer = Layer(x, f, y)
    assert layer.boxes_or_types == (x, f, y)
    assert not layer.is_plumbing
    assert layer.boxes == [f]
    assert Layer(f) == Layer(Ty(), f, Ty())
    assert Layer(permutation).boxes_or_types == (permutation, )
    assert Layer(x, swap, z).boxes == [Permutation(
        x @ x @ y @ z, [0, 2, 1, 3])]
    assert permutation == swap
    assert layer.dagger().dagger() == layer
    assert (z @ layer).boxes_or_types == (z @ x, f, y)
    assert (layer @ z).boxes_or_types == (x, f, y @ z)

    plumbed = Layer(x, f, permutation)
    assert plumbed.is_plumbing
    assert plumbed.boxes_or_types == (x, f, permutation)
    assert plumbed.boxes_and_offsets == [(f, 1), (permutation, 2)]
    assert plumbed.boxes == [f, permutation]
    assert plumbed.dagger() == Layer(x, f.dagger(), permutation.dagger())
    assert Layer(x, f).boxes_or_types == (x, f)
    with raises(ValueError):
        Layer(x)


def test_Layer_serialisation():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    permutation = Permutation(x @ y, [1, 0])
    for layer in (Layer(x, f, y), Layer(permutation), Layer(x, f, permutation)):
        for result in (
                loads(dumps(layer)), pickle.loads(pickle.dumps(layer))):
            assert result == layer
            assert result.boxes_or_types == layer.boxes_or_types
    identity = Permutation(x @ y, [0, 1])
    assert isinstance(loads(dumps(identity)), Permutation)


def test_Layer_identity_plumbing():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x, y)
    assert Layer(x, f, Permutation(y @ z, [0, 1])).boxes_or_types\
        == (x, f, y @ z)
    assert Layer(x, Permutation(y, [0]), f, z).boxes_or_types == (x @ y, f, z)
    assert Permutation(x @ y, [0, 1]).inside == ()
    with raises(ValueError):
        Layer(Permutation(x @ y, [0, 1]))


def test_Layer_coalesces_plumbing():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', z, z)
    permutation = Permutation(x @ y, [1, 0])
    assert Layer(x, permutation, y) == Layer(
        Permutation(x @ x @ y @ y, [0, 2, 1, 3]))
    assert Layer(permutation, f, permutation).boxes_or_types == (
        permutation, f, permutation)
    assert Layer(x, f, y, permutation, z).boxes_or_types == (
        x, f, Permutation(y @ x @ y @ z, [0, 2, 1, 3]))


def test_Layer_factory_ownership():
    from discopy import compact, markov, symmetric

    for module in (compact, markov):
        assert module.Diagram.permutation_factory is module.Permutation
        x, y = module.Ty('x'), module.Ty('y')
        permutation = module.Permutation(x @ y, [1, 0])
        layer = module.Layer(permutation)
        assert type(layer.boxes[0]) is module.Swap
        assert issubclass(module.Swap, module.Permutation)
        assert type(x @ permutation) is module.Permutation
    assert issubclass(markov.Layer, symmetric.Layer)
    assert not hasattr(symmetric.Layer, 'permutation_factory')


def test_Layer_tensor():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    left, right = Layer(x, f, y), Layer(z, g, x)
    result = left @ right
    assert result.boxes == [f, g]
    assert result.boxes_or_types == (x, f, y @ z, g, x)
    assert (left @ right).dagger() == left.dagger() @ right.dagger()
    permutation = Layer(Permutation(x @ y, [1, 0]))
    assert (left @ right) @ permutation == left @ (right @ permutation)
    assert (Layer(f) @ Layer(g) @ permutation).boxes_or_types == (
        f, g, permutation[0])
    plumbed = Layer(f, z) @ permutation @ Layer(x, g)
    assert plumbed.boxes_or_types == (f, z @ permutation[0] @ x, g)
    assert isinstance(plumbed[1], Permutation)


class Reiterable:
    """ An iterable that is neither an iterator nor a sequence. """
    def __init__(self, inside):
        self.inside = inside

    def __iter__(self):
        return iter(self.inside)


class Indexable:
    """ Iterable by the sequence protocol alone, with no ``__iter__``. """
    def __init__(self, items):
        self.items = list(items)

    def __getitem__(self, i):
        return self.items[i]


def test_Permutation_from_iterable():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    for dom, perm in [(x @ y, [1, 0]), (x @ y @ z, [1, 2, 0])]:
        listed = Permutation(dom, perm)
        for once in (iter(perm), (i for i in perm), dict.fromkeys(perm).keys(),
                     Reiterable(perm), Indexable(perm)):
            result = Permutation(dom, once)
            assert type(result) is type(listed) and result == listed
            assert hash(result) == hash(listed)
            namespace = {"monoidal": monoidal, "symmetric": symmetric}
            assert eval(repr(result), namespace) == result
    assert Swap(x @ y, iter([1, 0])) == Swap(x, y)


def test_noncommuting_Permutation_composition():
    x = Ty('x')
    first = Permutation(x ** 3, [1, 0, 2])
    second = Permutation(first.cod, [0, 2, 1])
    composite = first >> second
    assert Equation(composite, Permutation(x ** 3, [1, 2, 0]))
    assert not Equation(composite, Permutation(x ** 3, [2, 0, 1]))


def test_Permutation_box_setoid():
    x = Ty("x")
    p = Permutation(x ** 3, [1, 2, 0])
    q = p.dagger()
    f = Box("f", x ** 3, x ** 3)
    representatives = p, p[:], Id(p.dom) >> p, p >> Id(p.cod)
    for other in representatives:
        assert other == p and hash(other) == hash(p)
        assert other >> q == p >> q
        assert q >> other == q >> p
        assert Equation(other @ q, p @ q)
        assert other.dagger() == p.dagger()
    assert (p >> q) >> f == p >> (q >> f)
    assert Equation((p @ q) @ f, p @ (q @ f))


def test_permutation_factory():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    functor = Functor(ob_map={x: y, y: z, z: x}, ar_map={})
    assert functor(perm) == Permutation(y @ z @ x, [2, 0, 1])
    assert Equation(perm, perm.to_swaps())
    assert Equation(functor(perm), functor(perm.to_swaps()))


def test_Functor_on_whiskered_Swap():
    from discopy import balanced
    from discopy.matrix import Matrix
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    braided = Functor({x: x, y: y, z: z}, {}, cod=balanced.Diagram)
    assert braided(Swap(x, y) @ z) == balanced.Braid(x, y) @ z
    assert braided(Permutation(x @ y @ z, [2, 0, 1]))\
        == x @ balanced.Braid(y, z) >> balanced.Braid(x, z) @ y
    assert braided(Permutation(x @ y, [0, 1])) == balanced.Diagram.id(x @ y)
    assert [type(box) for box in (Swap(x, y) @ z).to_braided().boxes]\
        == [balanced.DualRailBraid]
    widen = Functor({Nat(1): x @ y}, {})
    assert Equation(
        widen(Diagram.swap(Nat(1), Nat(1)) @ Nat(1)),
        Diagram.swap(x @ y, x @ y) @ (x @ y))
    assert Functor({Nat(1): 2}, {}, cod=Matrix)(Permutation(
        Nat(3), [1, 0, 2])) == Matrix.permutation([1, 0, 2], [2, 2, 2])
    perm = Permutation(Nat(3), [2, 0, 1])
    assert Functor({Nat(1): Nat(1)}, {})(perm) == perm


def test_Functor_commutes_with_whiskering():
    from discopy import symmetric, markov, compact, frobenius
    for module in (symmetric, markov, compact, frobenius):
        x, y, z = map(module.Ty, "xyz")
        F = module.Functor(lambda ob: ob @ ob, lambda box: box,
                           dom=module.Diagram, cod=module.Diagram)
        assert F(module.Swap(x, y) @ z) == F(module.Swap(x, y)) @ F(z)
        assert F(module.Swap(x, y)) == module.Permutation(
            x @ x @ y @ y, [2, 3, 0, 1])
        assert module.Functor.id(module.Diagram)(module.Swap(x, y))\
            == module.Swap(x, y)


def test_Functor_id_on_Permutation():
    from discopy import symmetric, markov, compact, frobenius
    from discopy.matrix import Matrix
    for module in (symmetric, markov, compact, frobenius):
        x, y, z, w = map(module.Ty, "xyzw")
        perm = module.Permutation(x @ y @ z, [1, 2, 0])
        assert module.Functor.id(module.Diagram)(perm) == perm
        F = module.Functor({x: y @ z, y: w, z: x @ x}, {})
        staircase = module.Diagram.permutation(perm.perm, [y @ z, w, x @ x])
        assert F(perm) == module.Permutation(
            y @ z @ w @ x @ x, [2, 3, 4, 0, 1])
        assert module.Equation(F(perm), staircase)
        G = module.Functor({x: 2, y: 3, z: 1, w: 2}, {}, cod=Matrix)
        assert G(F(perm)) == G(staircase)


def test_inherited_permutation_factory():
    from discopy import closed, markov, frobenius, tensor

    cases = [
        (closed.markov, closed.markov.Ty("x"), closed.markov.Ty("y")),
        (markov, markov.Ty("x"), markov.Ty("y")),
        (frobenius, frobenius.Ty("x"), frobenius.Ty("y")),
        (tensor, tensor.Dim(2), tensor.Dim(3))]
    for module, x, y in cases:
        perm = module.Diagram.from_permutation([1, 0], x @ y)
        assert isinstance(perm, module.Diagram)
        assert perm.ar is module.Diagram
        assert all(isinstance(box, module.Swap) for box in perm.boxes)


def test_Permutation_whiskering():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm = Permutation(x @ y, [1, 0])
    assert perm @ z == Permutation(x @ y @ z, [1, 0, 2])
    assert z @ perm == Permutation(z @ x @ y, [0, 2, 1])
    assert isinstance(perm @ z, Permutation)
    assert isinstance(z @ perm, Permutation)
    assert perm @ z == perm @ Id(z)
    assert perm @ Ty() == perm == Ty() @ perm
    assert perm.tensor() == perm == perm.then()
    identity = Permutation(x, [0])
    assert isinstance(identity @ y, Permutation)
    assert isinstance(y @ identity, Permutation)


def test_mixed_Layer_plumbing():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    permutation = Permutation(x @ y, [1, 0])
    f = Box('f', z, z)
    layer = Layer(permutation, f, Ty())
    diagram = Diagram((layer,), layer.dom, layer.cod)
    assert diagram.dagger().dagger() == diagram
    assert diagram.to_hypergraph().dom == diagram.dom
    assert diagram.to_drawing().dom == diagram.dom.to_drawing()
    assert loads(dumps(diagram)) == diagram
    assert pickle.loads(pickle.dumps(diagram)) == diagram
    assert layer.boxes_and_offsets == [(permutation, 0), (f, 2)]
    assert diagram.boxes == [permutation, f]
    assert diagram.offsets == [0, 2]
    assert list(diagram.normalize()) == [diagram.to_staircases()]
    # A permutation is a box, so this layer holds two and cannot interchange.
    with raises(NotImplementedError):
        diagram.interchange(0, 0)
    g = Box('g', z, z)
    staircase = permutation @ z >> y @ x @ f
    assert staircase.substitute(1, g) == permutation @ z >> y @ x @ g
    assert staircase.substitute(
        0, Permutation(x @ y @ z, [1, 0, 2])) == staircase
    assert Diagram.decode(*staircase.encode()) == staircase


def test_unequal_arity_offsets():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y @ y), Box('g', z, Ty())
    permutation = Permutation(z @ x, [1, 0])
    layer = Layer(Ty(), f, permutation, g, x)
    assert layer.dom == x @ z @ x @ z @ x
    assert layer.cod == y @ y @ x @ z @ x
    assert layer.boxes_or_types == (f, permutation, g, x)
    assert layer.boxes_and_offsets == [(f, 0), (permutation, 2), (g, 4)]
    diagram = Diagram((layer,), layer.dom, layer.cod)
    assert diagram.offsets == [0, 2, 4]
    assert diagram.encode() == (
        layer.dom, [[(f, 0), (permutation, 2), (g, 4)]])
    assert Diagram.decode(*diagram.encode()) == diagram
    assert Diagram.decode(layer.dom, layer.boxes_and_offsets)\
        == diagram.to_staircases() != diagram


def test_Permutation_foliation():
    x, y, z, w = map(Ty, "xyzw")
    f0, f1 = Box("f0", w, x), Box("f1", z, y)
    g0, g1 = Box("g0", y, z), Box("g1", x, w)
    reverse = Permutation(x @ y @ z @ w, [3, 2, 1, 0])
    diagram = reverse >> f0 @ f1 @ g0 @ g1
    foliated = diagram.foliation()
    assert Equation(diagram, foliated)
    assert foliated.depth() == 1
    assert foliated.inside == (
        Layer(reverse), Layer(
            Ty(), f0, Ty(), f1, Ty(), g0, Ty(), g1, Ty()))
    assert foliated.boxes == [reverse, f0, f1, g0, g1]
    with raises(AxiomError):
        foliated.inside[0].merge(foliated.inside[1])
    inverse = reverse.dagger() >> reverse
    assert inverse.foliation() == Id(inverse.dom)
    assert (inverse >> reverse.dagger()).foliation() == reverse.dagger()
    assert (Swap(x, y) >> Swap(y, x)).foliation() == Id(x @ y)
    with raises(AxiomError):
        inverse.inside[0].merge(inverse.inside[1])


def test_large_Permutation_to_hypergraph():
    x, n = Ty("x"), 1100
    perm = Permutation(x ** n, reversed(range(n)))
    graph = perm.to_hypergraph()
    assert graph.boxes == ()
    assert graph.cod_wires == tuple(reversed(range(n)))


def test_default_Permutation_to_hypergraph():
    perm = Diagram.from_permutation([2, 1, 0])
    graph = perm.to_hypergraph()
    assert graph.cod == Nat(3)
    assert graph.cod_wires == (2, 1, 0)
    assert Equation(perm, perm.to_swaps())


def test_Functor_asks_the_level_of_its_codomain():
    from discopy.abc import SymmetricCategory
    from discopy.drawing import Drawing
    from discopy.python import Function
    from discopy.stream import Stream

    assert issubclass(Drawing, SymmetricCategory)
    assert issubclass(Stream, SymmetricCategory)

    x, y = Ty('x'), Ty('y')
    left, right = x.to_drawing(), y.to_drawing()
    assert Drawing.swap(left, right).cod == (y @ x).to_drawing()

    functor = Functor({x: int, y: str}, {}, cod=Function)
    cycle = functor(Permutation(x @ y @ x, [1, 2, 0]))
    assert cycle(0, "a", 1) == ("a", 1, 0)
    assert functor(Swap(x, y))(0, "a") == Function.swap(int, str)(0, "a")


def test_Permutation_to_drawing():
    from discopy.drawing import Drawing

    x, y, z = map(Ty, 'xyz')
    perm = Permutation(x @ y @ z, [1, 0, 2])
    functor = Functor(ob_map={
        x: z.to_drawing(), y: x.to_drawing(), z: y.to_drawing()},
        ar_map={}, cod=Drawing)
    drawing = functor(perm)
    assert drawing.dom == (z @ x @ y).to_drawing()
    assert drawing.cod == (x @ z @ y).to_drawing()


def test_coloured_Layer_merge():
    from discopy import monoidal

    red, green = map(monoidal.Colour, ("red", "green"))
    x = Ty(Wire('x', dom=red, cod=green))
    y = Ty(Wire('y', dom=green, cod=red))
    f, g = Box('f', x, x), Box('g', y, y)
    empty_red, empty_green = x[:0], x[len(x):]
    assert Layer(empty_red, f, empty_green) == Layer(f)
    assert Layer(f, y).merge(Layer(x, g)).boxes_or_types == (f, g)


def test_tensor_coalesces_plumbing():
    from functools import reduce
    x = Ty('x')
    perm = Permutation(x @ x, [1, 0])
    assert perm.tensor(x, perm) == Permutation(x ** 5, [1, 0, 2, 4, 3])
    assert perm.tensor() == perm
    f = Box('f', x, x)
    assert perm.tensor(f, perm) == reduce(
        lambda g, h: g.tensor(h), [perm, f, perm])
    assert perm.tensor(x, f) == reduce(lambda g, h: g.tensor(h), [perm, x, f])


def test_structural_boxes():
    from discopy.testable import assert_structural_boxes
    assert_structural_boxes(Diagram)


def test_term_reads_swaps_as_the_order_of_variables():
    x, y, z = map(Ty, "xyz")
    f, g = Box('f', x @ y, z), Box('g', x, y)
    term = (Swap(y, x) >> f).to_term()
    assert str(term) == "(y @ x)(lambda x0, x1: f(x1, x0))"
    assert term.eval() == Swap(y, x) >> f
    assert isinstance(term, Kappa) and term.category is Diagram
    diagram = Swap(x, x) >> x @ g >> Swap(x, y) >> Swap(y, x) >> f
    assert diagram.to_term().eval() == Swap(x, x) >> x @ g >> f
    assert Equation(diagram.to_term().eval(), diagram)
    with raises(ValueError):
        Swap(x, y).to_term()


def test_term_kappa_permutes():
    x, y, z = map(Ty, "xyz")
    k = Constant(Box('k', x @ y @ z, x))
    a, b, c = Variable('a', x), Variable('b', y), Variable('c', z)
    term = Kappa((c, a, b), k(a, b, c))
    assert term.stack == z @ x @ y and term.freevars == []
    assert term.eval() == Permutation(z @ x @ y, [1, 2, 0]) >> k.box
    same = Constant(Box('k', x @ x @ x, x))
    u, v, w = (Variable(name, x) for name in "uvw")
    cycle = Kappa((w, u, v), same(u, v, w)).eval()
    assert Equation(
        cycle, Diagram.permutation([1, 2, 0], x @ x @ x) >> same.box)
    assert not Equation(
        cycle, Diagram.permutation([2, 0, 1], x @ x @ x) >> same.box)
    planar = Kappa((a, b, c), k(a, b, c))
    assert planar.eval() == k.box
    assert planar == monoidal.Kappa((a, b, c), k(a, b, c))
    functor = Functor({x: y, y: z, z: x}, {k.box: Box('k', y @ z @ x, y)})
    assert term.eval(functor) == functor(term.eval())


def test_term_kappa_around_a_stack():
    x, y = Ty('x'), Ty('y')
    m, n = Constant(Box('m', x @ x @ y, x)), Constant(Box('n', x @ y @ x, x))
    a, c, v = Variable('a', x), Variable('c', x), Variable('v', y)
    with raises(ValueError):
        monoidal.Kappa((a, ), m(a, c))
    term = Kappa((a, ), m(a, c))
    assert term.freevars == [c] and term.offset == 1 and term.head == a
    assert term.stack == x @ y and term.dom == x @ x @ y
    assert term.eval() == Permutation(x @ x @ y, [1, 0, 2]) >> m.box
    body = Kappa((v, ), n(a, v, c))
    assert body.freevars == [a, c] and body.offset == 1 and body.stack == y
    behind = Kappa((c, ), body)
    assert not behind.front and behind.head is None
    assert behind.stack == y @ x and behind.eval() == n.box
    assert behind == monoidal.Kappa((c, ), body)
    ahead = Kappa((c, a), body)
    assert ahead.stack == y @ x @ x and ahead.head is None
    assert ahead.eval() == Permutation(y @ x @ x, [2, 0, 1]) >> n.box


def test_term_normal_form_with_exchange():
    x, y = Ty('x'), Ty('y')
    f, g = Constant(Box('f', x @ y, x)), Constant(Box('g', x, y))
    k = Constant(Box('k', x @ x @ x, x))
    a, b, c = (Variable(name, x) for name in "abc")
    term = x(lambda a: let(g(a), lambda v: f(b, v)))
    assert isinstance(term, Kappa) and term.eval().dom == x @ x
    assert term.normal_form() == x(lambda a: f(b, g(a)))
    assert TermBase.normal_form_soundness(term)
    stuck = Kappa((c, a), k(a, b, c))(f(Variable('u', x), Variable('v', y)))
    assert stuck.normal_form() == stuck
    assert stuck.freevars == [b, Variable('u', x), Variable('v', y)]
    assert TermBase.normal_form_soundness(stuck)
    effect = Constant(Box('e', x, Ty()))
    scalar = let(effect(a), lambda: k(b, c, Variable('d', x)), offset=1)
    assert scalar.freevars == [b, a, c, Variable('d', x)]
    assert scalar.eval() == x @ effect.box @ x @ x >> k.box
    assert str(Kappa((c, b), scalar)) == (
        "(x @ x)(lambda c, b: let(e(a), lambda: k(b, c, d), offset=1))")


def test_term_eval_into_a_braided_category():
    from discopy import balanced
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', y @ x @ x, x), balanced.Box('g', y @ x @ x, x)
    diagram = Permutation(x @ x @ y, [2, 0, 1]) >> f
    F = Functor({x: x, y: y}, {f: g}, cod=balanced.Diagram)
    expected = x @ balanced.Braid(x, y) >> balanced.Braid(x, y) @ x >> g
    assert F(diagram) == expected == diagram.to_term().eval(F)
    assert diagram.to_term().eval() == diagram


def test_term_errors_and_levels():
    x, y = Ty('x'), Ty('y')
    f = Constant(Box('f', x @ y, x))
    a, b = Variable('a', x), Variable('b', y)
    with raises(ValueError, match="distinct free variables of f.a, b., got"):
        Kappa((a, a), f(a, b))
    with raises(ValueError, match="distinct free variables"):
        Kappa((Variable('c', x), ), f(a, b))
    with raises(ValueError, match="offset 0 for the abstraction of b, a"):
        Kappa((b, a), f(a, b), 1)
    with raises(ValueError, match="Expected an offset"):
        Kappa((), f(a, b))
    with raises(ValueError, match="offset from 0 to 2"):
        Kappa((), f(a, b), 3)
    assert Kappa((), f(a, b), 1).freevars == [a, b]
    read = (y @ x)(lambda b, a: f(a, b))
    assert type(read) is Kappa and read == Kappa((b, a), f(a, b))
    low = monoidal.Constant(monoidal.Box('f', x @ y, x))
    with raises(ValueError, match="block of variables"):
        (y @ x)(lambda b, a: low(a, b))
    assert type(x(lambda a: a)) is monoidal.Kappa
    assert x(lambda a: a) == Kappa((a, ), a)
    assert type(f(a)) is Application and type(low(a)) is Application
    assert type(low(monoidal.Variable('a', x))) is monoidal.Application


def test_term_laws():
    x, y = Ty('x'), Ty('y')
    laws = (
        TermBase.eta, TermBase.reading, TermBase.normal_form_soundness,
        TermBase.normal_form_idempotence, TermBase.repr_transparency,
        TermBase.str_transparency, TermBase.serialisation)
    shapes = [
        [2, 1, 0, 1, 1], [1, 0, 0, 2, 2, 1, 2, 1, 1], [2, 2, 1, 2, 0, 1, 1],
        [1, 1, 1, 0, 1, 2, 0, 1], [2, 1, 2, 1, 1, 0, 1, 1, 0, 2, 0, 1]]
    swapped = 0
    for shape in shapes:
        term = TermBase.generate(y, shape, [x, y], "ab")
        assert all(law(term) for law in laws)
        swapped += any(
            isinstance(box, Permutation) for box in term.eval().boxes)
        renamed = TermBase.generate(y, shape, [x, y], "c")
        assert term.alpha_eq(renamed)
        assert Equation(term.eval(), renamed.eval())
    assert swapped
    sampler = Sampler(TermBase, iter([0, 1]), [x], "a")
    v, w = Variable('v', x), Variable('w', x)
    assert str(sampler.state(y, (v, w))) == "c0(w, v)"
