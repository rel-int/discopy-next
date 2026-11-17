# -*- coding: utf-8 -*-

"""
A feedforward network as a PyTorch module.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Module

Example
-------
>>> import torch  # doctest: +EXTRA
>>> from discopy.neural.network import Box, Dim, Dims, Network
>>> x, h = Dims(4), Dims(8)
>>> layer = Box('layer', x, h, module=torch.nn.Linear(4, 8))
>>> relu = Box('relu', h, h, module=torch.nn.ReLU())
>>> head = Box('head', h, x, module=torch.nn.Linear(8, 4))
>>> add = Box('add', x @ x, x, module=torch.add)
>>> residual = Network.copy(x) >> (layer >> relu >> head) @ x >> add
>>> module = Module(residual)
>>> module(torch.ones(5, 4)).shape
torch.Size([5, 4])
>>> len(list(module.parameters()))
4

The weights of a parametric network are parameters of the module, one for
each leg of its parameter space, so that a weight copied to two layers is
one parameter:

>>> from discopy import para
>>> w = Dims(Dim(4, 4))
>>> cell = para.Markov[Network](x, x, param=w, inside=Box(
...     'cell', x @ w, x, module=lambda v, W: torch.tanh(v @ W)))
>>> tied = (cell >> cell).reparam(Network.copy(w))
>>> module = Module(tied, torch.eye(4))
>>> module(torch.ones(5, 4)).shape
torch.Size([5, 4])
>>> [tuple(weight.shape) for weight in module.parameters()]
[(4, 4)]
"""

from __future__ import annotations

import torch

from discopy import para
from discopy.neural.network import Network


class Module(torch.nn.Module):
    """
    A network as a PyTorch module: the modules of its boxes are its
    submodules, so that their parameters are its parameters, and ``forward``
    runs the network on one tensor per input leg, returning one per output
    leg.

    Parameters:
        network : The feedforward network to run, or a parametric one, i.e.
            a :class:`discopy.para.Symmetric` over :class:`Network`.
        weights : The initial weights of a parametric network, one tensor
            per leg of its parameter space, each held as a parameter.

    Note
    ----
    A module shared by several boxes is registered once, so that the boxes
    share its weights. ``torch.compile``, ``copy.deepcopy`` and ``torch.save``
    take the module like any other, a copy running on its own parameters.
    """
    def __init__(self, network: Network | para.Symmetric,
                 *weights: torch.Tensor):
        super().__init__()
        self.network = network
        if len(weights) != len(getattr(network, "param", ())):
            raise ValueError(
                f"Expected one tensor per parameter leg, got {len(weights)}.")
        self.weights = torch.nn.ParameterList(weights)
        self.boxes = torch.nn.ModuleList({
            id(box.module): box.module for box in self.diagram.boxes
            if isinstance(box.module, torch.nn.Module)}.values())
        self.function = self.diagram.to_function(torch.Tensor)

    @property
    def diagram(self) -> Network:
        """ The network the module runs: the inside of a parametric one,
        from its inputs and its weights to its outputs. """
        if isinstance(self.network, para.Symmetric):
            return self.network.inside  # ty: ignore[invalid-return-type]
        return self.network

    def __getstate__(self):
        state = super().__getstate__()
        state.pop('function')
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.function = self.diagram.to_function(torch.Tensor)

    def forward(self, *tensors: torch.Tensor):
        return self.function(*tensors, *self.weights)
