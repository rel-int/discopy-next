from pytest import raises

from discopy.interaction import Diagram, Id, Ty
from discopy.monoidal import List
from discopy.utils import AxiomError


def test_Ty_repr():
    t = Ty[int](positive=1, negative=2)
    assert repr(t)\
        == str(t) == "interaction.Ty[int](positive=1, negative=2)"


def test_Ty_str():
    x, y, z, w = map(Ty, "xyzw")
    assert str(x @ -y @ z @ -w) == "x @ z @ -y @ -w"


def test_Ty_unit():
    assert Ty.unit() == Ty()
    assert Ty[int].unit() == Ty[int]() == Ty[int](0, 0)
    assert Ty[tuple].unit() == Ty[tuple]((), ())


def test_Ty_negatives():
    assert Ty.negatives is reversed
    x, y = Ty[List[int]]((1, ), (2, )), Ty[List[int]]((3, ), (4, ))
    assert x @ y == Ty[List[int]]((1, 3), (4, 2))
    with raises(TypeError):
        x @ Ty[int](1, 2)


def test_Diagram_permutation():
    from discopy import compact
    x0, x1, y0, y1, z0, z1 = map(
        compact.Ty, ("x0", "x1", "y0", "y1", "z0", "z1"))
    x, y, z = (
        Ty[compact.Ty](x0, x1),
        Ty[compact.Ty](y0, y1),
        Ty[compact.Ty](z0, z1))
    diagram = Diagram[compact.Diagram]
    permutation = diagram.permutation([2, 0, 1], [x, y, z])
    assert permutation.dom == x @ y @ z
    assert permutation.cod == z @ x @ y
    assert diagram.permutation([0, 1, 2], [x, y, z])\
        == diagram.id(x @ y @ z)
    with raises(ValueError):
        diagram.permutation([1, 0], [x, y, z])


def test_snake_equations():
    from discopy import symmetric
    x = Ty[symmetric.Ty](symmetric.Ty('a'), symmetric.Ty('b'))
    D = Diagram[symmetric.Diagram]
    hypergraph = lambda diagram: diagram.inside.to_hypergraph()
    left_snake = D.caps(x, -x) @ D.id(x) >> D.id(x) @ D.cups(-x, x)
    right_snake = D.id(x) @ D.caps(-x, x) >> D.cups(x, -x) @ D.id(x)
    assert hypergraph(left_snake) == hypergraph(D.id(x))
    assert hypergraph(right_snake) == hypergraph(D.id(x))


def test_ValueError():
    from discopy.ribbon import Ty as T, Diagram as D, Box as B
    x, y, z = map(Ty[T], "xyz")
    f = B('f', T('x'), T('y'))
    with raises(ValueError):
        Diagram[D](f, x, z)
    with raises(ValueError):
        Diagram[D](f, z, y)


def test_IndexError():
    with raises(IndexError):
        return Id()[:]


def test_Ty_split():
    from discopy import lens
    from discopy.symmetric import Ty as T
    x, x_, y, y_ = map(T, ["x", "x'", "y", "y'"])
    X, Y = Ty[T](x, x_), Ty[T](y, y_)
    assert (X @ Y).split() == (X, Y) == (X @ Y).split(left=True)
    assert (X @ Y).split(0) == (X @ Y, Ty[T]())
    assert (X @ Y).split(0, left=True) == (Ty[T](), X @ Y)
    assert (X @ Y).split((2, 1)) == (Ty[T](T(), x_), Ty[T](x @ y, y_))
    assert (X @ Y).split((1, 0), left=True) == (Ty[T](x), Ty[T](y, y_ @ x_))
    X, Y = lens.Ty(x, x_), lens.Ty(y, y_)
    assert (X @ Y).split() == (X, Y) == (X @ Y).split(left=True)
    assert not Ty[T]() and X and -X
    with raises(ValueError):
        (X @ Y).split(3)
    with raises(ValueError):
        (X @ Y).split((0, -1))


def test_trace():
    from discopy import symmetric
    from discopy.symmetric import Ty as T, Box
    a, a_, u, u_, b, b_ = map(T, ["a", "a'", "u", "u'", "b", "b'"])
    A, U, B = Ty[T](a, a_), Ty[T](u, u_), Ty[T](b, b_)
    D = Diagram[symmetric.Diagram]
    hypergraph = lambda diagram: diagram.inside.to_hypergraph()
    f = D(Box('f', a @ u @ u_ @ b_, b @ u @ u_ @ a_), A @ U, B @ U)
    feedback = A @ D.caps(U, -U) >> f @ -U >> B @ D.cups(U, -U)
    assert f.trace().dom == A and f.trace().cod == B
    assert hypergraph(f.trace()) == hypergraph(feedback)
    assert hypergraph(f.trace((1, 0)).trace((0, 1))) == hypergraph(f.trace())\
        == hypergraph(f.trace((0, 1)).trace((1, 0)))
    U_, A_, B_ = Ty[T](u), Ty[T](a, u_ @ a_), Ty[T](b, u_ @ b_)
    feedback = A_ @ D.caps(U_, -U_) >> f @ -U_ >> B_ @ D.cups(U_, -U_)
    assert hypergraph(f.trace((1, 0))) == hypergraph(feedback)
    g = D(Box('g', u @ a @ b_ @ u_, u @ b @ a_ @ u_), U @ A, U @ B)
    feedback = D.caps(-U, U) @ A >> -U @ g >> D.cups(-U, U) @ B
    assert hypergraph(g.trace(left=True)) == hypergraph(feedback)
    assert f.trace(0) == f and f.trace((0, 0), left=True) == f
    assert hypergraph(D.braid(U, U).trace()) == hypergraph(D.id(U))
    assert hypergraph(f.r.r) == hypergraph(f) == hypergraph(f.l.l)
    assert hypergraph(f.l) == hypergraph(f.r) == hypergraph(f.transpose())
    assert D.is_symmetric() and not Diagram.is_symmetric()
    with raises(AxiomError):
        f.trace(left=True)
    with raises(ValueError):
        f.trace(3)


def test_trace_Matrix():
    import numpy as np
    from discopy.abc import Nat
    from discopy.matrix import Matrix
    D, T = Diagram[Matrix[bool]], Ty[Nat]
    rng = np.random.default_rng(0)

    def diagram(dom, cod):
        shape = (len(dom.positive) + len(cod.negative),
                 len(cod.positive) + len(dom.negative))
        return D(Matrix[bool](rng.integers(0, 2, shape), *shape), dom, cod)

    A, U, B = T(Nat(1), Nat(2)), T(Nat(2), Nat(1)), T(Nat(2), Nat(1))
    f, g, n = diagram(A @ U, B @ U), diagram(U @ A, U @ B), (2, 1)
    assert f.trace(n) == A @ D.caps(U, -U) >> f @ -U >> B @ D.cups(U, -U)
    assert g.trace(n, left=True)\
        == D.caps(-U, U) @ A >> -U @ g >> D.cups(-U, U) @ B
    assert f.trace(n) == f.trace((0, 1)).trace((2, 0))\
        == f.trace((2, 0)).trace((0, 1))
    two = T(Nat(2), Nat(2))
    assert D.braid(two, two).trace(2) == D.id(two)\
        == D.braid(two, two).trace(2, left=True) == D.twist(two)
    h = diagram(A, B)
    assert h.r == h.transpose() and h.r.r == h == h.l.l
    assert h.conjugate() == h.r.dagger()
    assert h.curry().uncurry() == h == h.curry(left=False).uncurry(left=False)


def test_twist():
    from discopy.ribbon import Ty as T
    x = Ty(T('x'), T('u'))
    assert Diagram.twist(x) == Diagram.braid(x, x).trace()
    assert Diagram.twist(x).dom == x == Diagram.twist(x).cod
    assert Diagram.twist(x).inside.dom == T('x', 'u')


def test_curry():
    from discopy import symmetric
    from discopy.symmetric import Ty as T, Box
    x, x_, y, y_, z, z_ = map(T, ["x", "x'", "y", "y'", "z", "z'"])
    X, Y, Z = Ty[T](x, x_), Ty[T](y, y_), Ty[T](z, z_)
    D = Diagram[symmetric.Diagram]
    hypergraph = lambda diagram: diagram.inside.to_hypergraph()
    f = D(Box('f', x @ y @ z_, z @ y_ @ x_), X @ Y, Z)
    assert f.curry().dom == X and f.curry().cod == Z @ -Y
    assert f.curry(left=False).dom == Y and f.curry(left=False).cod == -X @ Z
    assert f.curry(0) == f == f.uncurry(0)
    for left in (True, False):
        assert hypergraph(f.curry(left=left).uncurry(left=left))\
            == hypergraph(f)
    assert hypergraph(f.curry((2, 0)).uncurry((2, 0))) == hypergraph(f)


def test_Function():
    from discopy import python
    from discopy.python import Function
    D, R = Diagram[Function], Ty[python.Ty]((float, ), (float, ))
    assert not hasattr(Function, "dagger") and not hasattr(Function, "twist")
    square = D(Function(
        lambda a, da: (a * a, 2 * a * da), (float, float), (float, float)),
        R, R)
    assert D.id(R).inside(1., 2.) == (1., 2.)
    assert (square @ square).inside(2., 3., 10., 100.) == (4., 9., 60., 400.)
    assert D.braid(R, R).inside(1., 2., 3., 4.) == (2., 1., 4., 3.)
    composite = square >> square >> D.id(R)
    assert composite.dom == R == composite.cod
    assert composite.inside.dom == R.positive @ R.negative\
        == composite.inside.cod
    assert composite.inside(2., 1.) == (16., 32.)


def test_additive_Function():
    """ Composition is the token machine of the geometry of interaction. """
    from discopy.python import additive
    X = additive.Function.ob(int)
    D, T = Diagram[additive.Function], Ty[additive.Function.ob]
    step = lambda forward: D(additive.Function(
        lambda m, n=0: (forward(m), 0) if n == 0 else (m, 1), X @ X, X @ X),
        T(X, X), T(X, X))
    f, g = step(lambda m: m + 1), step(lambda m: 2 * m)
    tokens = lambda diagram: [diagram.inside(5, i) for i in range(2)]
    assert tokens(f >> g) == [(12, 0), (5, 1)]
    assert tokens(D.id(T(X, X)) >> f) == tokens(f)\
        == tokens(f >> D.id(T(X, X)))


def test_Ty_pregroup():
    """ The negation of an integer type is both its adjoints, and it checks
    its cups and caps as a rigid type does. """
    from discopy.abc import Pregroup
    x, y = Ty('x'), Ty('y')
    assert isinstance(x, Pregroup) and x.l == x.r == -x
    assert x.is_adjoint(-x) and not x.is_adjoint(y)
    assert (x << y, x >> y) == (x @ -y, -x @ y)
    x.assert_isadjoint(-x)
    with raises(AxiomError):
        x.assert_isadjoint(y)
    with raises(AxiomError):
        Diagram.cups(x, y)
    with raises(TypeError):
        1 @ x


def test_trace_takes_the_keyword_form():
    from discopy import symmetric
    from discopy.symmetric import Ty as T, Box
    a, a_, u, u_, b, b_ = map(T, ["a", "a'", "u", "u'", "b", "b'"])
    A, U, B = Ty[T](a, a_), Ty[T](u, u_), Ty[T](b, b_)
    D = Diagram[symmetric.Diagram]
    f = D(Box('f', a @ u @ u_ @ b_, b @ u @ u_ @ a_), A @ U, B @ U)
    g = D(Box('g', u @ a @ b_ @ u_, u @ b @ a_ @ u_), U @ A, U @ B)
    assert f.trace() == f.trace(mem=U) == f.trace(cod=B)\
        == f.trace(dom=A, cod=B, mem=U) == f.feedback() == f.feedback(mem=U)
    assert f.trace((1, 0)) == f.trace(mem=Ty[T](u)) == f.feedback((1, 0))
    assert g.trace(left=True) == g.trace(mem=U, left=True)
    assert f.split_memory() == f.split_memory(mem=U) == (A, B, U)
    with raises(TypeError):
        f.trace(1, mem=U)
    with raises(AxiomError):
        f.trace(mem=A)
    with raises(AxiomError):
        f.trace(dom=B, mem=U)


def test_yanking():
    from discopy import compact
    from discopy.abc import Nat
    from discopy.matrix import Matrix
    x = Ty[Nat](Nat(1), Nat(2))
    assert Diagram[Matrix[bool]].yanking(x)
    X = Ty[compact.Ty](compact.Ty('x'), compact.Ty("x'", "y'"))
    assert Diagram[compact.Diagram].yanking.modulo(
        lambda diagram: diagram.inside.to_hypergraph())(X)
