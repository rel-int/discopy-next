# -*- coding: utf-8 -*-

import subprocess
import sys

from pytest import importorskip, raises

from discopy import neural, python
from discopy.neural.network import (
    Box, Cofunction, Copy, Dim, Dims, Discard, Equation, Function, Functor,
    Id, Lens, Merge, Network, Noise, Permutation, Swap, Trace, Unroll, Zeros,
    autodiff, tape, zeros)
from discopy.utils import AxiomError, dumps, loads


def test_lazy_torch_import():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import discopy.neural; "
        "assert 'torch' not in sys.modules"], check=True)


def test_dims():
    assert Dims(2, 3) == Dims(Dim(2), Dim(3)) == Dims(2) @ Dims(3)
    assert Dims(2, 3) @ Dims(4) == Dims(2, 3, 4) and Dims(2) ** 2 == Dims(2, 2)
    assert Dims() != Dims(1) == Dims(Dim()) and Dims(Dim(2, 3)) != Dims(2, 3)
    assert Dims(2, 3) * Dims(4, 5) == Dims(Dim(2, 4), Dim(2, 5), Dim(3, 4),
                                           Dim(3, 5))
    assert Dims(2) * Dims() == Dims() and Dims(2) * Dims(1) == Dims(2)
    assert Dims(2, 3)[1:] == Dims(3) and list(Dims(2, 3)) == [Dims(2), Dims(3)]
    assert repr(Dims(2, Dim(2, 3))) == str(Dims(2, Dim(2, 3)))\
        == "Dims(Dim(2), Dim(2, 3))"
    assert loads(dumps(Dims(2, Dim(2, 3)))) == Dims(2, Dim(2, 3))
    assert hash(Dims(2)) == hash(Dims(Dim(2)))
    with raises(TypeError):
        Dims('x')
    with raises(ValueError):
        Dims(0)
    with raises(TypeError):
        Dims(2) * Dim(2)


def test_distributivity_left_is_a_permutation():
    x, y, z = Dims(2, 3), Dims(4), Dims(5)
    left, right = x * (y @ z), (x * y) @ (x * z)
    assert left == Dims(Dim(2, 4), Dim(2, 5), Dim(3, 4), Dim(3, 5))
    assert right == Dims(Dim(2, 4), Dim(3, 4), Dim(2, 5), Dim(3, 5))
    assert left != right and left.legs() == right.legs()
    assert Dims.distributivity_left(x, y, z)
    assert not Dims.distributivity_left(x, y, z).modulo(None)
    assert Dims.distributivity_right(y, z, x).up_to is None


def test_check():
    import numpy as np
    Dims().check()
    Dims(2, Dim(2, 3), Dim()).check(
        np.ones((5, 2)), np.ones((5, 2, 3)), np.ones(()))
    with raises(AxiomError, match="Expected 2 tensors, got 1"):
        Dims(2, 3).check(np.ones(2))
    with raises(AxiomError, match="Expected a shape ending in"):
        Dims(Dim(2, 3)).check(np.ones((3, 2)))
    with raises(AxiomError):
        Dims(4).check(np.ones((4, 1)))


def test_box():
    f = Box('f', Dims(2), Dims(3, 3), module=lambda v: (v, v))
    assert f.module is f.data is f.dagger().module and f.module(1) == (1, 1)
    assert repr(f) == "neural.network.Box('f', Dims(Dim(2)), "\
        "Dims(Dim(3), Dim(3)))"
    assert repr(f.dagger()) == repr(f) + ".dagger()"
    assert Box('f', Dims(2), Dims(3, 3)) != f
    assert Box('f', Dims(2), Dims(3, 3), data=f.module) == f
    assert loads(dumps(f)) == Box('f', Dims(2), Dims(3, 3))
    one, other = (Box('f', Dims(2), Dims(2), module=object()) for _ in "12")
    assert one != other and one == one.dagger().dagger()
    scope = {"neural": neural, "Dims": Dims, "Dim": Dim}
    for box in (Box('f', Dims(2), Dims(3)), Box('f', Dims(2), Dims(3))[::-1]):
        assert eval(repr(box), scope) == box


def test_structure():
    x, y = Dims(2), Dims(3)
    assert Network.swap(x, y) == Swap(x, y) and Swap(x, y).dagger()\
        == Swap(y, x)
    assert Network.swap(Dims(), x) == Id(x)
    assert Network.from_permutation([2, 0, 1], x @ y @ x)\
        == Permutation(x @ y @ x, [2, 0, 1])
    assert Network.copy(x) == Copy(x) and Network.copy(x, 0) == Discard(x)
    assert Network.copy(x @ y) == Function(x @ y, [0, 1, 0, 1])
    assert Equation(
        Network.copy(x @ y), Copy(x) @ Copy(y) >> x @ Swap(x, y) @ y)
    assert Network.merge(x) == Merge(x) == Copy(x).dagger()
    assert Merge(x).dagger() == Copy(x) and Discard(x).dagger() == Noise(x)
    assert Network.noise(x) == Noise(x) == Merge(x, 0) == Discard(x).dagger()
    assert Network.merge(x @ y) == Cofunction(x @ y, [0, 1, 0, 1])
    assert Equation(
        Network.merge(x @ y), x @ Swap(y, x) @ y >> Merge(x) @ Merge(y))
    assert isinstance(Network.discard(x).boxes[0], Discard)
    assert isinstance(Network.merge(x).boxes[0], Merge)
    assert isinstance(Network.noise(x).boxes[0], Noise)
    for box in (Copy(x), Merge(x, 3), Discard(x), Noise(x)):
        assert loads(dumps(box)) == box
    with raises(ValueError):
        Copy(x @ y)
    with raises(ValueError, match="only spiders of neural.network.Network"):
        Network.spider_factory(2, 2, x)
    bialgebra = Merge(x) >> Copy(x)
    assert bialgebra.to_hypergraph().to_diagram() == bialgebra


def test_trace():
    x, h = Dims(2), Dims(4)
    cell = Box('cell', x @ h, x @ h)
    assert isinstance(cell.trace(), Trace)
    assert cell.trace().dom == cell.trace().cod == x
    assert cell.trace(left=True).dom == h and cell.trace(0) == cell
    assert cell.trace().dagger() == cell.dagger().trace()
    assert loads(dumps(cell.trace())) == cell.trace()
    with raises(AxiomError):
        Box('f', x, h).trace()
    assert not cell.trace().to_map().is_acyclic


def test_from_callable():
    x = Dims(4)
    layer, add = Box('layer', x, x), Box('add', x @ x, x)

    @Network.from_callable(x, x)
    def residual(v):
        return add(layer(v), v)

    assert residual == Network.copy(x) >> layer @ x >> add
    assert loads(dumps(residual)) == residual
    assert residual.to_hypergraph().boxes == (layer, add)


def test_equation():
    x = Dims(2)
    f, g = Box('f', x @ x, x @ x), Box('g', x, x)
    assert Equation((f >> x @ g).trace(), (x @ g >> f).trace())
    assert not Equation(f.trace(), f.trace(left=True))
    assert Equation(Swap(x, x).trace(), Id(x))


def test_functor():
    x = Dims(2)
    layer, add = Box('layer', x, x), Box('add', x @ x, x)
    residual = Network.copy(x) >> layer @ x >> add
    F = Functor(
        ob_map={x: (list, )},
        ar_map={layer: lambda v: [2 * i for i in v],
                add: lambda v, w: [i + j for i, j in zip(v, w)]},
        cod=python.Function)
    assert F(residual)([1, 2]) == [3, 6]
    assert F(x @ x) == python.Ty(list, list) and F(Dims()) == python.Ty()
    G = Functor(ob_map={x: x @ x},
                ar_map={layer: add >> add.dagger(), add: add @ add})
    assert G(residual)\
        == Network.copy(x @ x) >> (add >> add.dagger()) @ x @ x >> add @ add
    assert G(residual.trace()) == G(residual).trace(2)


def test_forward():
    import numpy as np
    x = Dims(2)
    split = Box('split', Dims(4), x @ x, module=lambda v: (v[:2], v[2:]))
    first, second = split.forward(np.arange(4))
    assert list(first) == [0, 1] and list(second) == [2, 3]
    assert Box('f', x, x, module=lambda v: v).forward(np.ones(2)).shape == (2,)
    with raises(ValueError, match="has no module"):
        Box('f', x, x).forward(np.ones(2))
    with raises(AxiomError, match="split: Expected 1 tensors, got 2"):
        split.forward(np.ones(2), np.ones(2))
    with raises(AxiomError, match="f: Expected a shape ending in"):
        Box('f', x, x, module=lambda v: v[:1]).forward(np.ones(2))


def test_to_function():
    import numpy as np
    x = Dims(2)
    double = Box('double', x, x, module=lambda v: 2 * v)
    add = Box('add', x @ x, x, module=lambda v, w: v + w)
    residual = Network.copy(x) >> double @ x >> add
    function = residual.to_function(np.ndarray)
    assert list(function(np.array([1, 2]))) == [3, 6]
    assert function.dom == function.cod == python.Function.ob(np.ndarray)
    permuted = (double @ x >> Swap(x, x) >> add).to_function()
    assert list(permuted(np.array([1, 2]), np.array([3, 4]))) == [5, 8]
    assert Network.discard(x).to_function()(np.ones(2)) == ()
    assert list(Id(x).to_function()(np.array([1, 2]))) == [1, 2]
    with raises(TypeError):
        function([1, 2])
    with raises(NotImplementedError, match="only feedforward"):
        (add >> Network.copy(x)).trace().to_function()
    with raises(ValueError, match="has no module"):
        Network.merge(x).to_function()
    with raises(NotImplementedError, match="no adjoint"):
        double.dagger().to_function()


def test_to_function_jax():
    jax = importorskip("jax")
    import jax.numpy as jnp
    x, h = Dims(2), Dims(3)

    def network(weights):
        linear = Box('linear', x, h, module=lambda v: v @ weights)
        relu = Box('relu', h, h, module=jax.nn.relu)
        add = Box('add', h @ h, h, module=jnp.add)
        return Network.copy(x) >> linear @ linear >> relu @ h >> add

    weights = jnp.ones((2, 3))
    function = jax.jit(network(weights).to_function(jax.Array))
    outputs = function(jnp.array([[1., -1.], [2., 2.]]))
    assert outputs.shape == (2, 3) and float(outputs[1, 0]) == 8.
    loss = lambda weights, inputs: jnp.sum(
        network(weights).to_function(jax.Array)(inputs))
    gradient = jax.grad(loss)(weights, jnp.ones((4, 2)))
    assert gradient.shape == (2, 3) and float(gradient[0, 0]) == 8.


def test_to_lens():
    import numpy as np
    x = Dims(2)
    square = Box('square', x, x, module=lambda v: v * v,
                 vjp=lambda v, dv: 2 * v * dv)
    add = Box('add', x @ x, x, module=lambda v, w: v + w,
              vjp=lambda v, w, dv: (dv, dv))
    v, dv = np.array([1, 2]), np.array([1, 10])
    residual = (Network.copy(x) >> square @ x >> add).to_lens(np.ndarray)
    assert isinstance(residual, Lens) and residual.dom == residual.cod
    assert residual.dom == Lens.ob((np.ndarray, ), (np.ndarray, ))
    assert list(residual.get(v)) == [2, 6]
    assert list(residual.put(v, dv)) == [3, 50]
    chain = (square >> square).to_lens()
    assert list(chain.get(v)) == [1, 16] and list(chain.put(v, dv)) == [4, 320]
    swapped = (square @ x >> Swap(x, x)).to_lens()
    assert [list(dw) for dw in swapped.put(v, v, dv, 0 * dv)] == [[0, 0], [1, 10]]
    dropped = (Network.copy(x) >> square @ Network.discard(x)).to_lens()
    assert list(dropped.put(v, dv)) == [2, 40]
    assert list(Network.discard(x).to_lens().put(v)) == [0, 0]
    assert list(Network.copy(x, 3).to_lens().put(v, dv, dv, dv)) == [3, 30]
    assert list(Id(x).to_lens().put(v, dv)) == list(dv)
    assert Network.copy(x @ x).to_lens().get(v, dv)[2] is v
    both = Network.copy(x @ x).to_lens().put(v, dv, v, dv, dv, v)
    assert [list(dw) for dw in both] == [[2, 12], [2, 12]]


def test_to_lens_errors():
    import numpy as np
    x = Dims(2)
    double = Box('double', x, x, module=lambda v: 2 * v)
    with raises(NotImplementedError, match="No reverse mode for numpy"):
        double.to_lens().put(np.ones(2), np.ones(2))
    with raises(NotImplementedError, match="only feedforward"):
        Box('f', x @ x, x @ x, module=lambda v, w: (v, w)).trace().to_lens()
    with raises(NotImplementedError, match="no adjoint"):
        double.dagger().to_lens()
    with raises(ValueError, match="has no module"):
        Box('f', x, x, vjp=lambda v, dv: dv).to_lens()
    with raises(ValueError, match="has no module"):
        Box('f', x, x).backward(np.ones(2), np.ones(2))
    wide = Box('f', x, x, module=lambda v: v, vjp=lambda v, dv: (dv, dv))
    with raises(AxiomError, match="f: Expected 1 tensors, got 2"):
        wide.backward(np.ones(2), np.ones(2))
    with raises(AxiomError, match="f: Expected a shape ending in"):
        wide.backward(np.ones(2), np.ones(3))
    assert autodiff(lambda: np.ones(2), (), (np.ones(2), )) == ()


def test_vjp_is_kept_but_not_compared():
    x = Dims(2)
    vjp = lambda v, dv: 2 * dv
    double = Box('double', x, x, module=None, vjp=vjp)
    assert double == Box('double', x, x) and double.vjp is vjp
    assert double.dagger().dagger().vjp is vjp
    assert repr(double) == repr(Box('double', x, x))
    assert 'vjp' not in double.to_tree() and loads(dumps(double)).vjp is None
    assert Copy(x).vjp is None and Swap(x, x).vjp is None


def random_network(rng, torch, width=3, depth=5):
    """ A feedforward network of linear, ReLU, add, copy and discard. """
    x = Dims(width)
    add = Box('add', x @ x, x, module=torch.add)
    relu = Box('relu', x, x, module=torch.relu)

    def linear():
        return Box('linear', x, x,
                   module=torch.nn.Linear(width, width).double())

    def step(kind):
        if kind == 'residual':
            return Network.copy(x) >> linear() @ x >> add
        if kind == 'fork':
            return Network.copy(x, 3) >> linear() @ (linear() >> relu) @ x\
                >> add @ x >> add
        if kind == 'drop':
            return Network.copy(x) >> linear() @ Network.discard(x)
        return relu if kind == 'relu' else linear()

    return Id(x).then(*(step(rng.choice(
        ['linear', 'relu', 'residual', 'fork', 'drop'])) for _ in range(depth)))


def test_to_lens_is_autograd():
    import random
    torch = importorskip("torch")
    rng = random.Random(0)
    torch.manual_seed(0)
    for _ in range(20):
        network = random_network(rng, torch)
        inputs = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        gradient = torch.randn(4, 3, dtype=torch.float64)
        outputs = network.to_function(torch.Tensor)(inputs)
        expected, = torch.autograd.grad(outputs, inputs, gradient)
        backprop = network.to_lens(torch.Tensor)
        assert torch.equal(backprop.get(inputs.detach()), outputs)
        assert torch.allclose(
            backprop.put(inputs.detach(), gradient), expected, atol=1e-6)


def test_to_lens_fails_when_copy_drops_a_branch(monkeypatch):
    torch = importorskip("torch")
    x = Dims(3)

    def copy(cls, pair, n=2):
        return cls(pair, pair ** n, cls.category.copy(pair.positive, n + 1),
                   cls.category(lambda *tensors: tensors[len(x)],
                                pair.positive @ (pair ** n).negative,
                                pair.negative), pair.positive)

    torch.manual_seed(0)
    linear = Box('linear', x, x, module=torch.nn.Linear(3, 3).double())
    network = Network.copy(x) >> linear @ x\
        >> Box('add', x @ x, x, module=torch.add)
    inputs = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    gradient = torch.randn(4, 3, dtype=torch.float64)
    expected, = torch.autograd.grad(
        network.to_function()(inputs), inputs, gradient)
    assert torch.allclose(network.to_lens().put(inputs.detach(), gradient),
                          expected, atol=1e-6)
    monkeypatch.setattr(Lens, "copy", classmethod(copy))
    assert not torch.allclose(
        network.to_lens().put(inputs.detach(), gradient), expected, atol=1e-6)


def test_to_lens_runs_each_layer_once():
    import numpy as np
    x, calls = Dims(2), []

    def layer(i):
        return Box(f'layer{i}', x, x,
                   module=lambda v: calls.append(i) or 2 * v,
                   vjp=lambda v, dv: 2 * dv)

    for depth in (1, 4, 16):
        calls.clear()
        chain = Id(x).then(*map(layer, range(depth))).to_lens(np.ndarray)
        assert list(chain.put(np.ones(2), np.ones(2))) == [2 ** depth] * 2
        assert calls == list(range(depth))


def test_to_lens_runs_each_module_once():
    torch = importorskip("torch")
    x, calls = Dims(2), []

    def module(v):
        calls.append(v)
        return torch.sin(v)

    chain = Id(x).then(*(Box(f'sin{i}', x, x, module=module)
                         for i in range(8))).to_lens(torch.Tensor)
    v = torch.tensor([.1, .2], dtype=torch.float64, requires_grad=True)
    dy = torch.ones(2, dtype=torch.float64)
    gradient = chain.put(v.detach(), dy)
    assert len(calls) == 8
    outputs = v
    for _ in range(8):
        outputs = torch.sin(outputs)
    expected, = torch.autograd.grad(outputs, v, dy)
    assert torch.allclose(gradient, expected)


def test_tape():
    import numpy as np
    jax = importorskip("jax")
    outputs, pullback = tape(jax.numpy.sin, (jax.numpy.zeros(2), ))
    assert list(outputs[0]) == [0, 0]
    assert list(pullback(jax.numpy.ones(2))[0]) == [1, 1]
    outputs, pullback = tape(lambda v: 2 * v, (np.ones(2), ))
    assert list(outputs[0]) == [2, 2]
    with raises(NotImplementedError, match="No reverse mode for numpy"):
        pullback(np.ones(2))
    assert tape(lambda: np.ones(2), ())[1](np.ones(2)) == ()


def test_autodiff_torch():
    torch = importorskip("torch")
    x = torch.tensor([1., 2.])
    assert torch.equal(autodiff(torch.relu, (-x, ), (x, ))[0], 0 * x)
    constant = autodiff(lambda v: torch.ones(2), (x, ), (x, ))
    assert torch.equal(constant[0], 0 * x)
    unused = autodiff(lambda v, w: (v * v, torch.ones(2)), (x, x), (x, x))
    assert torch.equal(unused[0], 2 * x * x) and torch.equal(unused[1], 0 * x)
    split = Box('split', Dims(2), Dims(1, 1),
                module=lambda v: (v[..., :1], v[..., 1:]))
    assert torch.equal(split.backward(x, x[:1], x[1:]), x)


def test_to_lens_jax():
    jax = importorskip("jax")
    import jax.numpy as jnp
    x, h = Dims(2), Dims(3)
    weights = jnp.arange(6.).reshape(2, 3)
    linear = Box('linear', x, h, module=lambda v: v @ weights)
    relu = Box('relu', h, h, module=jax.nn.relu)
    add = Box('add', h @ h, h, module=jnp.add)
    network = Network.copy(x) >> linear @ linear >> relu @ h >> add
    inputs, gradient = jnp.array([[1., -1.], [2., 2.]]), jnp.ones((2, 3))
    expected = jax.vjp(network.to_function(jax.Array), inputs)[1](gradient)[0]
    put = jax.jit(network.to_lens(jax.Array).put)
    assert jnp.allclose(put(inputs, gradient), expected)
    single = Box('single', x, x, module=lambda v: (2 * v, ))
    assert jnp.allclose(single.backward(inputs, inputs), 2 * inputs)
    split = Box('split', x, Dims(1, 1),
                module=lambda v: (v[..., :1], 3 * v[..., 1:]))
    halves = inputs[..., :1], inputs[..., 1:]
    assert jnp.allclose(
        split.backward(inputs, *halves), inputs * jnp.array([1., 3.]))


def test_unroll():
    import numpy as np
    x, m = Dims(2), Dims(Dim(2, 2), 3)
    add = Box('add', x @ x, x @ x, module=lambda v, w: (v + w, v + w),
              vjp=lambda v, w, dv, dw: (dv + dw, dv + dw))
    counter = add.trace()
    for rounds in (1, 2, 5):
        unrolled = counter.unroll(rounds)
        assert unrolled == Unroll(rounds)(counter)
        assert unrolled.dom == unrolled.cod == x
        assert not any(isinstance(box, Trace) for box in unrolled.boxes)
        assert [box.name for box in unrolled.boxes].count('add') == rounds
        v = np.array([1, 2])
        assert list(counter.to_function(np.ndarray, rounds)(v)) == [
            rounds, 2 * rounds]
        assert list(counter.to_lens(rounds=rounds).put(v, 3 * v)) == [
            3 * rounds, 6 * rounds]
    left = Box('add', x @ x, x @ x, module=add.module).trace(left=True)
    assert list(left.to_function(rounds=3)(np.array([1, 2]))) == [3, 6]
    assert left.unroll(1).boxes[1].module == Zeros(x, x)
    swap = Swap(x, x).trace()
    assert list(swap.to_function(rounds=1)(np.array([1, 2]))) == [0, 0]
    assert list(swap.to_function(rounds=2)(np.array([1, 2]))) == [1, 2]
    nested = (add @ x >> x @ add).trace().trace().unroll(2)
    assert [box.name for box in nested.boxes].count('add') == 2 * 2 * 2
    assert Id(x).unroll(3) == Id(x) and add.unroll(3) == add
    wide = Zeros(x, m)(np.ones((5, 4, 2), dtype=int))
    assert [w.shape for w in wide] == [(5, 4, 2, 2), (5, 4, 3)]
    assert wide[0].dtype == int and not wide[0].any()
    with raises(ValueError, match="at least one round, got 0"):
        counter.unroll(0)
    with raises(NotImplementedError, match="no input"):
        Box('f', x, x, module=lambda v: v).trace().to_function(rounds=1)()


def recurrent(torch, width=3, state=4):
    """ A recurrent cell and a readout of its state, with a trace. """
    x, m, y = Dims(width), Dims(state), Dims(2)
    linear = torch.nn.Linear(width + state, state).double()
    cell = Box('cell', x @ m, m, module=lambda v, h: torch.tanh(
        linear(torch.cat([v, h], -1))))
    readout = Box('readout', m, y, module=torch.nn.Linear(state, 2).double())
    return (cell >> Network.copy(m) >> readout @ m).trace()


def test_backprop_through_time():
    torch = importorskip("torch")
    torch.manual_seed(0)
    network = recurrent(torch)
    cell, _, readout = network.boxes[0].arg.boxes
    for rounds in (1, 2, 5):
        inputs = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        gradient = torch.randn(6, 2, dtype=torch.float64)
        state = torch.zeros(6, 4, dtype=torch.float64)
        for _ in range(rounds):
            state = cell.module(inputs, state)
        outputs = readout.module(state)
        expected, = torch.autograd.grad(outputs, inputs, gradient)
        function = network.to_function(torch.Tensor, rounds)
        assert torch.equal(function(inputs), outputs)
        backprop = network.to_lens(torch.Tensor, rounds)
        assert torch.equal(backprop.get(inputs.detach()), outputs)
        assert torch.allclose(
            backprop.put(inputs.detach(), gradient), expected, atol=1e-6)


def test_reverse_preserves_the_rounds_of_a_trace():
    torch = importorskip("torch")
    torch.manual_seed(0)
    network = recurrent(torch)
    body = network.boxes[0].arg.to_lens(torch.Tensor)
    for rounds in (1, 2, 5):
        inputs = torch.randn(6, 3, dtype=torch.float64)
        gradient = torch.randn(6, 2, dtype=torch.float64)
        states = [zeros(inputs, (6, 4))]
        for _ in range(rounds):
            outputs, state = body.get(inputs, states[-1])
            states.append(state)
        total, back = 0 * inputs, (gradient, 0 * states[-1])
        for i, state in enumerate(reversed(states[:-1])):
            step, back = body.put(inputs, state, *back)
            total, back = total + step, (0 * gradient, back)
        unrolled = network.to_lens(torch.Tensor, rounds)
        assert torch.allclose(unrolled.get(inputs), outputs, atol=1e-12)
        assert torch.allclose(
            unrolled.put(inputs, gradient), total, atol=1e-12)


def test_unroll_jax():
    jax = importorskip("jax")
    import jax.numpy as jnp
    x = Dims(2)
    cell = Box('cell', x @ x, x @ x, module=lambda v, h: (
        jnp.tanh(v + h), jnp.tanh(v + h)))
    network = cell.trace()
    inputs, gradient = jnp.array([[1., -1.], [.5, 2.]]), jnp.ones((2, 2))
    function = network.to_function(jax.Array, rounds=3)
    assert jnp.allclose(function(inputs), jnp.tanh(
        inputs + jnp.tanh(inputs + jnp.tanh(inputs))))
    expected = jax.vjp(function, inputs)[1](gradient)[0]
    put = jax.jit(network.to_lens(jax.Array, rounds=3).put)
    assert jnp.allclose(put(inputs, gradient), expected)


def test_delay_is_the_identity():
    f = Box('f', Dims(2), Dims(3)) >> Copy(Dims(3))
    assert Dims(2).delay() == Dims(2) and f.delay() == f
