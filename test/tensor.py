import pytest

pytest.importorskip("tensornetwork")

import numpy as np
import tensornetwork as tn
from hypothesis import find
from pytest import raises

from discopy.utils import AxiomError, dumps, loads
from discopy.tensor import *
from discopy import frobenius


def test_backend():
    import jax.numpy
    import torch
    assert isinstance(Tensor.id().array, np.ndarray)
    with backend('jax'):
        assert isinstance(Tensor.id().array, jax.numpy.ndarray)
        with backend('pytorch'):
            assert isinstance(Tensor.id().array, torch.Tensor)
        assert isinstance(Tensor.id().array, jax.numpy.ndarray)
    assert isinstance(Tensor.id().array, np.ndarray)


def test_Dim():
    with raises(TypeError):
        Dim('a')
    with raises(ValueError):
        Dim(-1)
    dim = Dim(2, 3)
    assert Dim(1) @ dim == dim @ Dim(1) == dim
    assert Dim(1).tensor(*(Dim(2, 3), Dim(4), Dim(1))) == Dim(2, 3, 4)
    assert dim[:1] == Dim(3, 2)[1:] == Dim(2)
    assert dim[0] == Dim(3, 2)[1]
    assert dim.inside[0] == 2
    assert repr(Dim(1, 2, 3)) == str(dim) == "Dim(2, 3)"
    assert {dim: 42}[dim] == 42
    assert Dim(2, 3, 4).r == Dim(4, 3, 2)
    assert loads(dumps(dim)) == dim and loads(dumps(Dim())) == Dim() == Dim(1)


def test_Dim_unwind():
    f = Box('f', Dim(2), Dim(2), [1, 0, 0, 1])
    assert Dim(2).unwind() == Dim(2)
    assert f.to_hypergraph().to_diagram() == f
    assert (f @ f).foliation().inside == (Diagram.layer_factory(f, f), )
    assert (f >> f).simplify() == f >> f


def test_Tensor():
    assert Tensor([1], Dim(1), Dim(1))
    m = Tensor([0, 1, 1, 0], Dim(2), Dim(2))
    assert repr(m) == str(m)\
        == "Tensor[int64]([0, 1, 1, 0], dom=Dim(2), cod=Dim(2))"
    u = Tensor([1, 0, 0, 0], Dim(2), Dim(2))
    v = Tensor([0, 0, 0, 1], Dim(2), Dim(2))
    assert u + v == Tensor.id(Dim(2))
    with raises(TypeError):
        u + [0, 0, 0, 1]
    with raises(AxiomError):
        u + u @ Tensor([1, 0], Dim(1), Dim(2))
    with raises(TypeError):
        u >> Dim(2)
    arr = np.array([1, 0, 0, 1, 0, 1, 1, 0]).reshape((2, 2, 2))
    m = Tensor(arr, Dim(2, 2), Dim(2))
    assert m == m and np.all(m.array == arr)
    m = Tensor([0, 1, 1, 0], Dim(2), Dim(2))
    assert Tensor.id(Dim(2)).then(*(m, m)) == m >> m.dagger()


def test_Spider_array():
    d = Dim(2)
    assert (Spider(1, 2, d).array == Tensor.spiders(1, 2, d).array).all()
    with raises(NotImplementedError):
        Spider(1, 1, d, .5).eval()
    with raises(NotImplementedError):
        (Spider(1, 2, d) >> Spider(2, 1, d, .5)).to_tn()


def test_Spider_to_tn():
    d = Dim(2)
    tensor = Spider(1, 1, d) >> Spider(1, 2, d) >> Spider(2, 0, d)
    result = tn.contractors.auto(*tensor.to_tn()).tensor
    assert all(result == np.array([1, 1]))


def test_Spider_to_tn_pytorch():
    try:
        with backend('pytorch') as np:
            tn.set_default_backend('pytorch')
            from torch import float64

            d = Dim(2)

            alice = Box[float64]("Alice", Dim(1), d,
                        np.array([1., 2.]).requires_grad_(True))
            tensor = alice >> Spider[float64](1, 2, d) >> \
                     Spider[float64](2, 0, d)
            result = tn.contractors.auto(*tensor.to_tn()).tensor
            assert result.item() == 3
    finally:
        tn.set_default_backend('numpy')


def test_Tensor_cups():
    assert np.all(Tensor.cups(Dim(2), Dim(2)).array == np.identity(2))
    with raises(TypeError):
        Tensor.cups(Dim(2), 2)
    with raises(TypeError):
        Tensor.cups(2, Dim(2))
    with raises(AxiomError):
        Tensor.cups(Dim(3), Dim(2))


def test_Tensor_caps():
    assert np.all(Tensor.caps(Dim(2), Dim(2)).array == np.identity(2))
    with raises(TypeError):
        Tensor.caps(Dim(2), 2)
    with raises(TypeError):
        Tensor.caps(2, Dim(2))
    with raises(AxiomError):
        Tensor.caps(Dim(3), Dim(2))


def test_Tensor_transpose():
    assert Tensor.caps(Dim(2), Dim(2)).transpose()\
        == Tensor.cups(Dim(2), Dim(2))


def test_Tensor_conjugate():
    assert Tensor[complex]([1j], Dim(1), Dim(1)).conjugate().array == -1j


def test_Tensor_tensor():
    assert Tensor.tensor(Tensor.id(Dim(2))) == Tensor.id(Dim(2))

    assert Tensor.id(Dim(2)) @ Tensor.id(Dim(3)) == Tensor.id(Dim(2, 3))

    v = Tensor([1, 0], Dim(1), Dim(2))
    assert v @ v == Tensor([1, 0, 0, 0], dom=Dim(1), cod=Dim(2, 2))
    assert v @ v.dagger() == v << v.dagger()

    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f, g = frobenius.Box('f', x, x), frobenius.Box('g', y, y)
    ob, ar = {x: 2, y: 3}, {f: [1, 0, 0, 1], g: list(range(9))}
    F = Functor(ob, ar)
    assert F(f) @ F(g) == F(f @ g)


def test_tensor_swap():
    f = Tensor([1, 0, 0, 1], Dim(2), Dim(2))
    g = Tensor(list(range(9)), Dim(3), Dim(3))
    swap = Tensor.swap(Dim(2), Dim(3))
    assert f @ g >> swap == swap >> g @ f


def test_tensor_spiders():
    with raises(NotImplementedError):
        Tensor.spiders(1, 2, Dim(3), [0.5])


def test_Functor_repr():
    x = frobenius.Ty('x')
    F = Functor({x: 2}, {}, dom=frobenius.Diagram, dtype=bool)
    assert repr(F) ==\
        "tensor.Functor(ob_map={frobenius.Ty(frobenius.Wire('x')): 2}, "\
        "ar_map={}, dom=frobenius.Diagram, dtype=bool)"


def test_Functor_call():
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f, g = frobenius.Box('f', x @ x, y), frobenius.Box('g', y, frobenius.Ty())
    ob = {x: 2, y: 3}
    ar = {f: list(range(2 * 2 * 3)), g: list(range(3))}
    F = Functor(ob, ar)
    assert list(F(f >> g).array.flatten()) == [5.0, 14.0, 23.0, 32.0]
    assert list(F(g.transpose(left=True)).array.flatten()) == [0.0, 1.0, 2.0]
    with raises(TypeError):
        F("Alice")
    assert Functor(ob_map={x: Dim(2, 3)}, ar_map=None)(x) == Dim(2, 3)


def test_Functor_call_opt_einsum_missing(monkeypatch):
    """ opt_einsum is not a declared dependency: contracting past
    ``MAX_EINSUM_INDICES`` without it installed should raise a helpful
    ``ImportError`` rather than crash with no explanation. """
    import sys
    import discopy.config
    monkeypatch.setattr(discopy.config, "MAX_EINSUM_INDICES", 0)
    monkeypatch.setitem(sys.modules, "opt_einsum", None)
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f, g = frobenius.Box('f', x @ x, y), frobenius.Box('g', y, frobenius.Ty())
    F = Functor({x: 2, y: 3}, {f: list(range(2 * 2 * 3)), g: list(range(3))})
    with raises(ImportError, match="opt_einsum"):
        F(f >> g)


def test_Functor_call_opt_einsum_other_import_error(monkeypatch):
    """ A broken opt_einsum install failing on an unrelated import must not
    be misdiagnosed as opt_einsum being absent. """
    import builtins
    import discopy.config
    monkeypatch.setattr(discopy.config, "MAX_EINSUM_INDICES", 0)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "opt_einsum":
            raise ImportError("broken", name="not_opt_einsum")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f, g = frobenius.Box('f', x @ x, y), frobenius.Box('g', y, frobenius.Ty())
    F = Functor({x: 2, y: 3}, {f: list(range(2 * 2 * 3)), g: list(range(3))})
    with raises(ImportError, match="broken"):
        F(f >> g)


def test_Functor_swap():
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f, g = frobenius.Box('f', x, x), frobenius.Box('g', y, y)
    F = Functor({x: 2, y: 3}, {f: [1, 2, 3, 4], g: list(range(9))})
    assert F(f @ g >> frobenius.Swap(x, y)) == \
           F(frobenius.Swap(x, y) >> g @ f)


def test_AxiomError():
    m = Tensor([1, 0, 0, 1, 0, 1, 1, 0], Dim(2, 2), Dim(2))
    with raises(AxiomError) as err:
        m >> m


def test_Functor_sum():
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f = frobenius.Box('f', x, y)
    F = Functor({x: 1, y: 2}, {f: [1, 0]})
    assert F(f + f) == F(f) + F(f)


def test_Tensor_radd():
    m = Tensor([1, 0, 0, 1, 0, 1, 1, 0], Dim(2, 2), Dim(2))
    assert 0 + m == m


def test_Tensor_iter():
    v = Tensor([0, 1], Dim(1), Dim(2))
    assert list(v) == [0, 1]
    s = Tensor([1], Dim(1), Dim(1))
    with raises(TypeError):
        # how does one iterate over a scalar?
        list(s)


def test_Tensor_subs():
    import sympy
    from sympy.abc import x
    s = Tensor[sympy.Expr]([x], Dim(1), Dim(1))
    assert s.subs(x, 1).array == 1


def test_Diagram_cups_and_caps():
    with raises(AxiomError):
        Diagram.cups(Dim(2), Dim(3))


def test_Diagram_swap():
    x, y, z = Dim(2), Dim(3), Dim(4)
    assert Diagram.swap(x, y @ z) == \
        (Swap(x, y) @ Id(z)) >> (Id(y) @ Swap(x, z))


def test_Box():
    f = Box[int]('f', Dim(2), Dim(2), [0, 1, 1, 0])
    assert repr(f) == "tensor.Box[int]('f', Dim(2), Dim(2), data=[0, 1, 1, 0])"
    assert {f: 42}[f] == 42


def test_Spider():
    assert repr(Spider(1, 2, Dim(3))) == "tensor.Spider(1, 2, Dim(3))"
    assert Spider(1, 2, Dim(2)).dagger() == Spider(2, 1, Dim(2))
    with raises(ValueError):
        Spider(1, 2, Dim(2, 3))


def test_Swap_to_tn():
    nodes, order = Swap(Dim(2), Dim(2)).to_tn()
    assert order == [nodes[0][0], nodes[1][0], nodes[1][1], nodes[0][1]]


def test_Tensor_scalar():
    s = Tensor([1], Dim(1), Dim(1))
    for ptype in [int, float, complex]:
        assert isinstance(ptype(s), ptype)


def test_rotated_Box_eval():
    f = Box("f", Dim(2), Dim(2), [1, 2, 3, 4])
    assert f.r.eval().is_close(f.transpose().eval())
    assert f.l.eval().is_close(f.transpose(left=True).eval())


def test_Tensor_adjoint_eval():
    alice = Box("Alice", Dim(1), Dim(2), [1, 2])
    eats = Box("eats", Dim(1), Dim(2, 3, 2), [3] * 12)
    food = Box("food", Dim(1), Dim(2), [4, 5])

    diagram = alice @ eats @ food >>\
        Cup(Dim(2), Dim(2)) @ Id(Dim(3)) @ Cup(Dim(2), Dim(2))

    tensor1 = diagram.eval()
    tensor2 = diagram.transpose_box(2).transpose_box(0, left=True).eval()
    assert tensor1 == tensor2


def test_Tensor_dtype_inference():
    assert Box("F(A)", Dim(1), Dim(1), data=None).dtype is None
    assert Box("X", Dim(1), Dim(1), data=[0]) == Box[np.int64]("X", Dim(1), Dim(1), data=[0])
    assert Box("Y", Dim(1), Dim(1), data=[1.]) == Box[np.float64]("Y", Dim(1), Dim(1), data=[1.])
    assert Box("Y", Dim(1), Dim(1), data=[1]) != Box("Y", Dim(1), Dim(1), data=[1.])


def test_non_numpy_eval():
    reference = Swap(Dim(2), Dim(2)).eval()
    with backend('pytorch'):
        result = Swap(Dim(2), Dim(2)).eval()
    assert np.allclose(np.asarray(result.array), reference.array)


def test_eval_over_52_indices():
    f = Box('f', Dim(2), Dim(2), [1, 0, 0, 1])
    diagram = f
    for _ in range(60):
        diagram = diagram >> f
    assert np.allclose(diagram.eval().array, np.eye(2))
    g = Box('g', Dim(2), Dim(2), [1., 0., 0., 1.])
    diagram = g
    for _ in range(60):
        diagram = diagram >> g
    with backend('pytorch'):
        result = diagram.eval()
    assert np.allclose(np.asarray(result.array), np.eye(2))


def test_Tensor_array():
    box = Box("box", Dim(2), Dim(2), None)
    assert box.array is None


def test_CMap_eval():
    vector = Box('vector', Dim(1), Dim(2), [0., 1.])
    f = Box('f', Dim(2), Dim(2), [0., 1., 1., 0.])
    a = Box('a', Dim(1), Dim(2), [1., 2.])
    b = Box('b', Dim(1), Dim(2), [3., 4.])
    u = Box('u', Dim(1), Dim(2, 3), list(map(float, range(6))))
    spider = Spider(1, 1, Dim(2)) >> Spider(1, 2, Dim(2)) \
        >> Spider(2, 0, Dim(2))
    for diagram in [
            vector >> vector[::-1],
            spider,
            f >> f,
            a @ b >> Swap(Dim(2), Dim(2)),
            u >> u[::-1],
            Id(Dim(2)),
            Cup(Dim(2), Dim(2)),
            Cap(Dim(2), Dim(2)) >> Cup(Dim(2), Dim(2))]:
        tensor, reference = diagram.to_map().eval(), diagram.eval()
        assert (tensor.dom, tensor.cod) == (reference.dom, reference.cod)
        assert np.allclose(
            np.asarray(tensor.array, dtype=float),
            np.asarray(reference.array, dtype=float))


def test_CMap_eval_jax():
    import jax
    import jax.numpy as jnp
    with backend('jax'):
        b = lambda x: Box[float]('v', Dim(1), Dim(2), x * jnp.ones(2))
        f = lambda x: (b(x) >> b(x)[::-1]).to_map().eval().array
        assert jax.grad(f)(1.) == 4.
        assert float(jax.jit(f)(1.)) == 2.


def test_Functor_backend():
    x = frobenius.Ty('x')
    f = frobenius.Box('f', x, x @ x)
    diagram = f >> frobenius.Spider(1, 1, x) @ x >> f.dagger()
    ob, ar = {x: 2}, {f: [1., 2., 3., 4., 5., 6., 7., 8.]}
    reference = Functor(ob, ar)(diagram)
    with backend('jax'):
        with_backend = Functor(ob, ar)(diagram)
    assert np.allclose(np.asarray(with_backend.array), reference.array)


def test_Functor_markov_copy():
    from discopy import markov
    n = markov.Ty('n')
    f = markov.Box('f', n, n)
    F = Functor({n: Dim(2)}, {f: [1., 2., 3., 4.]}, dom=markov.Diagram)
    reference = F(f) >> Tensor.copy(Dim(2), 2)
    assert F(f >> markov.Copy(n, 2)).is_close(reference)


def test_Functor_pytorch():
    import torch
    from torch import float64
    x = frobenius.Ty('x')
    v = frobenius.Box('v', frobenius.Ty(), x)
    t = torch.tensor(3.0, dtype=float64, requires_grad=True)
    F = Functor({x: 2}, {v: torch.stack([t, t])}, dtype=float64)
    with backend('pytorch'):
        result = F(v >> v.dagger()).array
    result.backward()
    assert result.item() == 18. and t.grad.item() == 12.


def test_spider_eval_pytorch():
    import torch
    with backend('pytorch'):
        result = (Spider(0, 1, Dim(2)) >> Spider(1, 0, Dim(2))).eval()
        assert isinstance(result.array, torch.Tensor)
        assert result.array.item() == 2.


def test_eval_params():
    vector = Box('vector', Dim(1), Dim(2), [1., 2.])
    diagram = vector >> vector[::-1]
    assert diagram.eval(optimize="optimal") == diagram.eval()
    assert diagram.eval(order='C') == diagram.eval()
    x = frobenius.Ty('x')
    v = frobenius.Box('v', frobenius.Ty(), x)
    F = Functor({x: 2}, {v: [1., 2.]}, optimize="optimal")
    assert F(v >> v.dagger()).array == 5.
    assert Functor({x: 2}, {v: [1., 2.]}, order='C')(v >> v.dagger()) \
        == F(v >> v.dagger())
    assert repr(F) == \
        "tensor.Functor(ob_map={frobenius.Ty(frobenius.Wire('x')): 2}, " \
        "ar_map={frobenius.Box('v', frobenius.Ty(), " \
        "frobenius.Ty(frobenius.Wire('x'))): [1.0, 2.0]}, " \
        "dom=frobenius.Diagram, dtype=float, optimize='optimal')"


def test_Functor_bubble():
    men = Box("men", Dim(1), Dim(2), [0, 1])
    mortal = Box("mortal", Dim(2), Dim(1), [1, 1])
    men_are_mortal = (men >> mortal.bubble()).bubble()
    assert men_are_mortal.eval(dtype=bool)
    F = Functor(lambda x: x, lambda box: box.array, dom=Diagram)
    assert np.allclose(
        np.asarray(F(men_are_mortal.arg).array, dtype=float),
        np.asarray(men_are_mortal.arg.eval().array, dtype=float))


def test_then_promotes_dtype():
    """ The integer structural tensors compose with floats as units. """
    g = Tensor([[.5, .5], [.25, .75]], Dim(2), Dim(2))
    assert Tensor.id(Dim(2)) >> g == g == g >> Tensor.id(Dim(2))
    assert Tensor.id(Dim(1)) @ g == g
    assert Tensor.spiders(1, 1, Dim(2)) >> g == g
    assert Tensor.swap(Dim(2), Dim(2)) >> g @ g\
        == g @ g >> Tensor.swap(Dim(2), Dim(2))
    copies = Tensor.copy(Dim(2)) >> g @ g
    assert copies.dtype == g.dtype and np.allclose(
        copies.array[0], np.outer(g.array[0], g.array[0]))
    c = Tensor([[1j, 0], [0, 1]], Dim(2), Dim(2))
    assert Tensor.id(Dim(2)) >> c == c
    assert Tensor[float].id(Dim(2)) >> Tensor.id(Dim(2))\
        == Tensor[float].id(Dim(2))


def test_Tensor_is_markov():
    """ Tensor has copy and discard, so a lens over it needs no discard. """
    from discopy import lens
    from discopy.abc import MarkovCategory
    assert issubclass(Tensor, MarkovCategory)
    g = Tensor([[.5, .5], [.25, .75]], Dim(2), Dim(2))
    lifted = lens.Markov[Tensor].lift(g)
    assert lifted.forward == Tensor.copy(Dim(2)) >> g @ Tensor.id(Dim(2))
    assert lifted.put == Tensor.discard(Dim(2)) @ Tensor.id(Dim(1))
    assert lifted.forward.dtype == g.dtype


def test_Tensor_then_unbiased():
    """ Composing several tensors folds Tensor.then, not Matrix.then. """
    t = Tensor(list(range(16)), Dim(2, 2), Dim(2, 2))
    assert t.then() == t
    with raises(TypeError):
        t.then(None)
    with raises(TypeError):
        t.then(None, t)
    assert t.then(t, t) == t >> t >> t
    u = Tensor(list(range(36)), Dim(2, 3), Dim(2, 3))
    assert u.then(u, u) == u >> u >> u


def test_Tensor_comonoid():
    """ The spiders are a commutative comonoid under the Kronecker product. """
    assert Tensor.copy(Dim(2)) == Tensor(
        [1, 0, 0, 0, 0, 0, 0, 1], Dim(2), Dim(2, 2))
    for x in (Dim(1), Dim(2), Dim(3), Dim(2, 2)):
        assert Tensor.copy_coassociativity(x)
        assert Tensor.copy_cocommutativity(x)
        assert Tensor.copy_counitality(x)
        assert Tensor.copy_coherence(x, Dim(3))


def test_Tensor_ar():
    """ A subscript of Tensor is its own factory, so its swaps keep dtype. """
    assert Tensor[complex].ar is Tensor[complex]
    assert Tensor.ar is Tensor and Box[complex].ar is Diagram
    assert Swap[complex](Dim(2), Dim(3)).eval().dtype == complex
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    F = Functor({x: 2, y: 3}, {}, dtype=complex)
    swap = frobenius.Swap(x, y)
    assert F(swap).dtype == F(swap >> swap.dagger()).dtype == complex
    assert F(frobenius.Permutation(x @ y @ x, [1, 2, 0])).dtype == complex


def test_Tensor_strategy():
    tensor = find(Tensor.strategy(), lambda t: len(t.dom) == 2 and t.cod)
    assert isinstance(tensor, Tensor[int])
    assert tensor.array.shape == tensor.dom.inside + tensor.cod.inside


def test_delay_is_the_identity():
    f = Box('f', Dim(2), Dim(2), [1, 0, 0, 1])
    diagram = f @ Dim(3) >> Spider(1, 2, Dim(2)) @ Dim(3)\
        >> Dim(2) @ Swap(Dim(2), Dim(3))
    assert Dim(2).delay() == Dim(2) == Dim(2).head == Dim(2).tail
    for d in (f, diagram):
        assert d.delay() == d == d.delay(2) and d.delay().eval() == d.eval()


def test_feedback_is_the_trace():
    array = np.arange(36).reshape(2, 3, 2, 3)
    f = Box('f', Dim(2, 3), Dim(2, 3), array.flatten())
    right = np.einsum('aibi->ab', array)
    for traced in (f.feedback(), f.feedback(1), f.feedback(mem=Dim(3)),
                   f.feedback(dom=Dim(2), cod=Dim(2), mem=Dim(3)),
                   f.trace(), f.trace(1), f.trace(mem=Dim(3))):
        assert isinstance(traced, Diagram)
        assert np.array_equal(traced.eval().array, right)
    left = np.einsum('iaib->ab', array)
    for traced in (f.feedback(left=True), f.trace(mem=Dim(2), left=True)):
        assert np.array_equal(traced.eval().array, left)


def test_spider_with_no_legs_is_the_circle():
    for d in [Dim(2), Dim(3), Dim(2, 3)]:
        circle = Tensor.spiders(0, 1, d) >> Tensor.spiders(1, 0, d)
        assert Tensor.spiders(0, 0, d) == circle


def test_Box_dagger_is_not_the_box():
    t = Box('t', Dim(2), Dim(2), data=[1, 2, 3, 4])
    assert t.dagger() != t and t.dagger().dagger() == t
    assert hash(t.dagger()) != hash(t)


def test_Tensor_trace():
    f = Tensor[int](list(range(64)), Dim(2, 2, 2), Dim(2, 2, 2))
    x, y = Dim(2), Dim(2, 2)
    feedback = y @ Tensor[int].caps(x, x) >> f @ x >> y @ Tensor[int].cups(x, x)
    assert f.trace() == f.trace(mem=x) == feedback
    assert f.trace(0) == f and f.trace(3).dom == f.trace(3).cod == Dim(1)
    with raises(AxiomError):
        Tensor[int]([0, 1, 2, 3, 4, 5], Dim(2), Dim(3)).trace()
    with raises(ValueError):
        Tensor[int]([0, 1, 2, 3], Dim(2), Dim(2)).trace(2)
    g = Tensor[int](list(range(8)), Dim(2, 2), Dim(2))
    assert g.trace() == Tensor[int]([0 + 3, 4 + 7], Dim(2), Dim(1))
    assert g.trace(left=True) == Tensor[int]([0 + 5, 2 + 7], Dim(2), Dim(1))


def test_Tensor_repeat():
    swap = Tensor[bool].swap(Dim(2), Dim(2))
    assert swap.repeat() == swap + Tensor[bool].id(Dim(2, 2))
    with raises(TypeError):
        Tensor[int].swap(Dim(2), Dim(2)).repeat()
    with raises(TypeError):
        Tensor[bool]([1, 0, 0, 1, 1, 0], Dim(2), Dim(3)).repeat()


def test_spider_with_no_legs_in_every_dtype():
    for dtype, circle in [(bool, True), (int, 2), (float, 2.), (complex, 2)]:
        assert Tensor[dtype].spiders(0, 0, Dim(2)).array == circle


def test_to_tn_state_spider():
    for diagram in [Spider(0, 1, Dim(2)) @ Id(Dim(3)), Spider(0, 2, Dim(2))]:
        nodes, order = diagram.to_tn()
        array = tn.contractors.auto(nodes, output_edge_order=order).tensor
        assert array.tolist() == diagram.eval().array.tolist()


def test_Functor_no_dtype_repr_and_algebraic_conjugate():
    from discopy.quantum.channel import Channel, Q
    F = Functor({frobenius.Ty('x'): 2}, {}, dtype=None)
    assert "dtype=None" in repr(F)
    conjugate = Channel.id(Q(Dim(2))).conjugate(diagrammatic=False)
    assert isinstance(conjugate, Channel)
