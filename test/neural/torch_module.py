# -*- coding: utf-8 -*-

import copy
import io

from pytest import importorskip, mark, raises

from discopy.neural.network import Box, Dim, Dims, Network
from discopy.utils import AxiomError


def test_module():
    torch = importorskip("torch")
    from discopy.neural.torch import Module
    x, h = Dims(4), Dims(8)
    layer = Box('layer', x, h, module=torch.nn.Linear(4, 8))
    relu = Box('relu', h, h, module=torch.nn.ReLU())
    head = Box('head', h, x, module=torch.nn.Linear(8, 4))
    add = Box('add', x @ x, x, module=torch.add)
    residual = Network.copy(x) >> (layer >> relu >> head) @ x >> add
    module = Module(residual)
    inputs = torch.ones(5, 4)
    assert module(inputs).shape == (5, 4) and module.network is residual
    assert list(module.boxes) == [layer.module, relu.module, head.module]
    assert len(list(module.parameters())) == 4
    module(inputs).sum().backward()
    assert all(p.grad is not None for p in module.parameters())
    twice = Module(layer >> head >> layer >> head)
    assert len(list(twice.boxes)) == 2 == len(list(twice.parameters())) // 2
    outputs = Module(layer >> Box('split', h, x @ x, module=lambda v: (
        v[:, :4], v[:, 4:])))(inputs)
    assert tuple(output.shape for output in outputs) == ((5, 4), (5, 4))
    with raises(AxiomError, match="layer: Expected a shape ending in"):
        module(torch.ones(5, 3))
    with raises(TypeError):
        module([1.] * 4)



def parametric(torch, tied):
    """ Two tanh layers with their weights as parameters, tied or not. """
    from discopy import para
    x, w = Dims(3), Dims(Dim(3, 3))
    cell = para.Markov[Network](x, x, param=w, inside=Box(
        'cell', x @ w, x, module=lambda v, W: torch.tanh(v @ W)))
    return (cell >> cell).reparam(Network.copy(w)) if tied else cell >> cell


@mark.parametrize("tied", [False, True])
def test_descent_is_a_step_of_sgd(tied):
    torch = importorskip("torch")
    from discopy.neural.network import Learner, Lens
    from discopy.neural.torch import Module
    torch.manual_seed(0)
    network, rate = parametric(torch, tied), 0.1
    initial = [torch.randn(3, 3, dtype=torch.float64)
               for _ in range(len(network.param))]
    inputs, gradient = (
        torch.randn(5, 3, dtype=torch.float64) for _ in range(2))
    module = Module(network, *(weight.clone() for weight in initial))
    assert len(list(module.parameters())) == (1 if tied else 2)
    optimiser = torch.optim.SGD(module.parameters(), lr=rate)
    (module(inputs) * gradient).sum().backward()
    optimiser.step()
    learner = Learner.reverse(network, torch.Tensor)
    assert learner.param == Lens.ob((torch.Tensor, ), (torch.Tensor, ))\
        ** len(initial)
    _, *gradients = learner.inside.put(inputs, *initial, gradient)
    _, *updated = learner.descent(rate).inside.put(inputs, *initial, gradient)
    for weight, new, grad, old in zip(
            module.parameters(), updated, gradients, initial):
        assert torch.allclose(weight.grad, grad, atol=1e-12)
        assert torch.allclose(weight.detach(), new, atol=1e-12)
        assert not torch.allclose(new, old)


def test_tying_commutes_with_backprop():
    torch = importorskip("torch")
    from discopy.neural.network import Learner, Lens
    torch.manual_seed(0)
    weight, inputs, gradient = (
        torch.randn(3, 3, dtype=torch.float64) for _ in range(3))
    before = Learner.reverse(parametric(torch, True), torch.Tensor)
    untied = Learner.reverse(parametric(torch, False), torch.Tensor)
    after = untied.reparam(Lens.copy(before.param))
    assert before.param == after.param
    for left, right in zip(before.inside.put(inputs, weight, gradient),
                           after.inside.put(inputs, weight, gradient)):
        assert torch.allclose(left, right, atol=1e-12)


def test_parametric_module():
    torch = importorskip("torch")
    from discopy.neural.torch import Module
    network = parametric(torch, True)
    module = Module(network, torch.eye(3))
    assert module.diagram is network.inside
    assert torch.equal(module(torch.ones(2, 3)),
                       torch.tanh(torch.tanh(torch.ones(2, 3))))
    clone = copy.deepcopy(module)
    assert torch.equal(clone(torch.ones(2, 3)), module(torch.ones(2, 3)))
    assert clone.weights[0] is not module.weights[0]
    with raises(ValueError, match="one tensor per parameter leg, got 2"):
        Module(network, torch.eye(3), torch.eye(3))
    with raises(ValueError, match="one tensor per parameter leg, got 1"):
        Module(network.inside, torch.eye(3))
    with raises(AxiomError, match="cell: Expected a shape ending in"):
        Module(network, torch.eye(2))(torch.ones(2, 3))


def test_module_unhashable():
    torch = importorskip("torch")
    from discopy.neural.torch import Module

    class Scaled(torch.nn.Linear):
        def __eq__(self, other):
            return isinstance(other, Scaled)

    x = Dims(2)
    first, second = Scaled(2, 2), Scaled(2, 2)
    module = Module(Box('first', x, x, module=first)
                    >> Box('second', x, x, module=second))
    assert list(module.boxes) == [first, second]


@mark.slow
def test_compile():
    torch = importorskip("torch")
    from discopy.neural.torch import Module
    x = Dims(3)
    layer = Box('layer', x, x, module=torch.nn.Linear(3, 3))
    add = Box('add', x @ x, x, module=torch.add)
    module = Module(Network.copy(x) >> layer @ x >> add)
    inputs = torch.rand(2, 3)
    assert torch.allclose(torch.compile(module)(inputs), module(inputs))


def test_copy_runs_its_own_parameters():
    torch = importorskip("torch")
    from discopy.neural.torch import Module
    x = Dims(3)
    layer = Box('layer', x, x, module=torch.nn.Linear(3, 3))
    add = Box('add', x @ x, x, module=torch.add)
    module = Module(Network.copy(x) >> layer @ x >> layer @ x >> add)
    inputs = torch.rand(2, 3)
    expected = module(inputs)
    twin = copy.deepcopy(module)
    with torch.no_grad():
        for parameter in twin.parameters():
            parameter.zero_()
    assert torch.equal(twin(inputs), inputs)
    assert torch.equal(module(inputs), expected)
    twin(inputs).sum().backward()
    assert all(p.grad is not None for p in twin.parameters())
    assert all(p.grad is None for p in module.parameters())
    buffer = io.BytesIO()
    torch.save(module, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert torch.equal(loaded(inputs), expected)
    with torch.no_grad():
        for parameter in loaded.parameters():
            parameter.zero_()
    assert torch.equal(loaded(inputs), inputs)
    assert torch.equal(module(inputs), expected)
