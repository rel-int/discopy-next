""" The types and the gradient check shared by the tests of
:mod:`discopy.neural.datatype` and :mod:`discopy.neural.machine`. """
from discopy import para, python
from discopy.markov import Diagram, Ty

S, P = Ty("S"), Ty("P")
Int = python.Ty(int)
Para, Net = para.Markov[Diagram], para.Markov[python.Function]


def grad(p):
    """ The gradient of a tensor, zero when it was not reached. """
    import torch
    return torch.zeros_like(p) if p.grad is None else p.grad


def check_grads(module, reference):
    """ The weights of ``module`` have the gradients of ``reference``. """
    import torch
    for weight, expected in zip(module.weights, reference):
        assert (weight.grad is None) == (expected.grad is None)
        assert torch.allclose(grad(weight), grad(expected), atol=1e-6)
