# -*- coding: utf-8 -*-

"""
Equivariant layers as weight tying: a linear map between permutation
representations of a finite group commutes with the group exactly when its
matrix is constant on the orbits of the group on pairs of points.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Dense
    Legs

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        tying
        act
        layer
        equivariance

Example
-------
The linear maps on three points that commute with rotations are the
circulant matrices, one weight for each orbit of the rotations on pairs of
points:

>>> import torch  # doctest: +EXTRA
>>> from discopy.group import Cyclic
>>> from discopy.neural.torch import Module
>>> rotations = Cyclic[3].action()
>>> tying(rotations, rotations).tolist()
[[0, 1, 2], [2, 0, 1], [1, 2, 0]]
>>> shift = torch.tensor([0., 1., 0.])
>>> module = Module(layer(rotations, rotations), shift)
>>> module(torch.tensor([[1., 2., 4.]])).tolist()
[[2.0, 4.0, 1.0]]
>>> dense, v = Dense(rotations, rotations), torch.tensor([[[1.], [2.], [4.]]])
>>> equivariance(lambda v: dense(v, shift.reshape(3, 1, 1)),
...              rotations, rotations, v)
0.0

Its legs are those of :class:`discopy.neural.network.Network`, with the
unit dimensions dropped, see :class:`Legs`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy
import torch

from discopy import para
from discopy.group import Action
from discopy.neural.network import Box, Dim, Dims, Network
from discopy.python.finset import Permutation


def tying(dom: Action, cod: Action) -> numpy.ndarray:
    """
    The orbit of each pair of points, i.e. which weight of an equivariant
    linear map from ``dom`` to ``cod`` each of its entries is tied to: the
    array of shape ``(cod.size, dom.size)`` whose entry ``[y, x]`` is the
    number of the orbit of the pair ``(y, x)`` among the orbits of the tensor
    of the actions, see :meth:`Action.orbits`.

    Parameters:
        dom : The action on the input points, the columns.
        cod : The action on the output points, the rows.

    Raises:
        TypeError : If the actions are of different groups.

    Example
    -------
    The symmetric group has two orbits on pairs of points, the diagonal and
    the rest, and its invariant maps send a set to a sum over its points:

    >>> from discopy.group import Cyclic, Symmetric
    >>> action = Symmetric[3].action()
    >>> tying(action, action).tolist()
    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    >>> tying(action, Action.trivial(Symmetric[3])).tolist()
    [[0, 0, 0]]
    >>> tying(Cyclic[3].action(), Cyclic[2].action())  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    TypeError: Expected an action of ...
    """
    index = numpy.empty(cod.size * dom.size, dtype=int)
    for k, orbit in enumerate(cod.tensor(dom).orbits()):
        index[list(orbit)] = k
    return index.reshape(cod.size, dom.size)


def act(image: Permutation, tensor: torch.Tensor) -> torch.Tensor:
    """
    The action of a permutation on a tensor whose second-to-last axis runs
    over the points: ``t[..., image[x], :] = tensor[..., x, :]``.

    Parameters:
        image : The permutation, e.g. the image of a generator of an
            :class:`Action`.
        tensor : A tensor of shape ``(..., size, channels)``.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> v = torch.tensor([[10.], [20.], [30.]])
    >>> act(Permutation((1, 2, 0)), v).flatten().tolist()
    [30.0, 10.0, 20.0]
    """
    return tensor[..., list(image.dagger()), :]


class Dense(torch.nn.Module):
    """
    The dense realisation of the equivariant linear maps from ``dom`` to
    ``cod`` with channels, the group acting trivially on them: one
    ``(D, C)`` matrix of weights for each orbit of the group on pairs of
    points, and one bias vector of size ``D`` for each orbit on output points,

    ``out[..., y, d] = sum(w[o(y, x), d, c] * v[..., x, c]) + b[o(y), d]``.

    It materialises a ``(|Y|, |X|, D, C)`` weight, in ``O(|Y||X|DC)`` memory:
    it is a dense reference, not meant to scale, fast realisations come
    separately.

    Parameters:
        dom : The action on the input points.
        cod : The action on the output points.

    Attributes:
        orbits : The number of orbits on pairs of points, i.e. of weights.
        bias_orbits : The number of orbits on output points, i.e. of biases.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Cyclic
    >>> rotations = Cyclic[3].action()
    >>> dense = Dense(rotations, rotations)
    >>> dense.orbits, dense.bias_orbits
    (3, 1)
    >>> w = torch.tensor([[[0.]], [[1.]], [[0.]]])
    >>> dense(torch.tensor([[1.], [2.], [4.]]), w).flatten().tolist()
    [2.0, 4.0, 1.0]
    """
    def __init__(self, dom: Action, cod: Action):
        super().__init__()
        self.dom, self.cod = dom, cod
        index = tying(dom, cod)
        bias = tying(Action.trivial(cod.group), cod).reshape(-1)
        self.orbits = int(index.max()) + 1 if index.size else 0
        self.bias_orbits = int(bias.max()) + 1 if bias.size else 0
        self.register_buffer(
            "index", torch.from_numpy(index), persistent=False)
        self.register_buffer(
            "bias_index", torch.from_numpy(bias), persistent=False)

    def forward(self, v: torch.Tensor, w: torch.Tensor,
                b: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply the layer.

        Parameters:
            v : The input, of shape ``(..., |X|, C)``.
            w : The weights, of shape ``(orbits, D, C)``.
            b : The biases, of shape ``(bias_orbits, D)``, if any.

        Returns:
            The output, of shape ``(..., |Y|, D)``.
        """
        out = torch.einsum(
            "yxdc,...xc->...yd",
            w[self.index],  # ty: ignore[invalid-argument-type]
            v)
        if b is None:
            return out
        return out + b[self.bias_index]  # ty: ignore[invalid-argument-type]


class Legs(Dense):
    """
    A :class:`Dense` layer on the tensors of the legs of a box, i.e. with
    the unit dimensions dropped: with one input channel the points are the
    last axis, with one output channel so are the output points, and a
    single point has no axis at all. The weights and the biases are legs too,
    so their unit dimensions are dropped as well.

    Parameters:
        dom : The action on the input points.
        cod : The action on the output points.
        channels : The number of input and output channels ``(C, D)``.

    Attributes:
        channels : The number of input and output channels ``(C, D)``.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Cyclic
    >>> rotations = Cyclic[3].action()
    >>> legs = Legs(rotations, rotations)
    >>> legs(torch.tensor([1., 2., 4.]), torch.tensor([0., 1., 0.]))
    tensor([2., 4., 1.])
    """
    def __init__(self, dom: Action, cod: Action,
                 channels: tuple[int, int] = (1, 1)):
        super().__init__(dom, cod)
        self.channels = channels

    @staticmethod
    def restore(tensor: torch.Tensor, *shape: int) -> torch.Tensor:
        """
        Restore the unit dimensions that a leg of :class:`Dims` drops: the
        trailing axes of a tensor of the leg ``Dim(*shape)`` are reshaped to
        ``shape``.

        Parameters:
            tensor : A tensor whose trailing axes are the leg ``Dim(*shape)``.
            shape : The full shape of the leg.

        Example
        -------
        >>> import torch  # doctest: +EXTRA
        >>> Legs.restore(torch.zeros(5, 3), 3, 1).shape
        torch.Size([5, 3, 1])
        >>> Legs.restore(torch.zeros(5), 1, 1).shape
        torch.Size([5, 1, 1])
        """
        kept = tensor.ndim - len(Dim(*shape).inside)
        return tensor.reshape((*tensor.shape[:kept], *shape))

    def forward(self, v: torch.Tensor, w: torch.Tensor,
                b: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply the layer to the tensors of the legs.

        Parameters:
            v : The input, of shape ``(..., |X|, C)`` without unit axes.
            w : The weights, of shape ``(orbits, D, C)`` without unit axes.
            b : The biases, of shape ``(bias_orbits, D)`` without unit axes.

        Returns:
            The output, of shape ``(..., |Y|, D)`` without unit axes.
        """
        c, d = self.channels
        out = super().forward(
            self.restore(v, self.dom.size, c),
            self.restore(w, self.orbits, d, c),
            None if b is None else self.restore(b, self.bias_orbits, d))
        return out.reshape((*out.shape[:-2], *Dim(self.cod.size, d).inside))


def layer(dom: Action, cod: Action, channels: tuple[int, int] = (1, 1),
          bias: bool = False, name: str = "W") -> para.Markov:
    """
    An equivariant layer as a parametric map, from the points of ``dom``
    with ``C`` channels each to those of ``cod`` with ``D`` channels each,
    whose parameter is its weights ``(orbits, D, C)`` and, with a bias, its
    biases ``(bias_orbits, D)``, see :class:`Dense`.

    Parameters:
        dom : The action on the input points.
        cod : The action on the output points.
        channels : The number of input and output channels ``(C, D)``.
        bias : Whether the layer has a bias.
        name : The name of its box.

    Example
    -------
    >>> from discopy.group import Symmetric
    >>> action = Symmetric[4].action()
    >>> deep_sets = layer(action, action, (3, 2), bias=True)
    >>> deep_sets.dom == Dims(Dim(4, 3)) and deep_sets.cod == Dims(Dim(4, 2))
    True
    >>> [dim.inside for dim in deep_sets.param.inside]
    [(2, 2, 3), (2,)]

    Unit dimensions are dropped from the legs, so that with one channel in
    and out the legs are the points and the weights, one for each orbit:

    >>> layer(action, action).param
    Dims(Dim(2))
    """
    c, d = channels
    module = Legs(dom, cod, channels)
    param = Dims(Dim(module.orbits, d, c))
    if bias:
        param = param @ Dims(Dim(module.bias_orbits, d))
    x, y = Dims(Dim(dom.size, c)), Dims(Dim(cod.size, d))
    box = Box(name, x @ param, y, module=module)
    return para.Markov[Network](x, y, param=param, inside=box)


def equivariance(function: Callable[[torch.Tensor], torch.Tensor],
                 dom: Action, cod: Action, *tensors: torch.Tensor) -> float:
    """
    How far a function is from commuting with the group: the largest
    ``|f(g.v) - g.f(v)|`` over the generators ``g`` and the tensors ``v``,
    where ``g`` acts by :func:`act` through ``dom`` on the input and through
    ``cod`` on the output. It is zero, up to rounding, exactly when the
    function commutes with every generator, hence with the group.

    Parameters:
        function : A function of one tensor of points, its other arguments,
            e.g. its weights, closed over.
        dom : The action on the input points.
        cod : The action on the output points.
        tensors : The inputs to test on, of shape ``(..., |X|, C)``.

    Raises:
        TypeError : If the actions are of different groups.

    Example
    -------
    A circulant matrix commutes with the rotations, another does not:

    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Cyclic
    >>> rotations = Cyclic[3].action()
    >>> v = torch.tensor([[1.], [2.], [4.]])
    >>> circulant = torch.tensor([[1., 2, 3], [3, 1, 2], [2, 3, 1]])
    >>> equivariance(lambda v: circulant @ v, rotations, rotations, v)
    0.0
    >>> equivariance(lambda v: v.flip(0), rotations, rotations, v)
    3.0
    """
    if dom.group is not cod.group:
        raise TypeError(f"Expected actions of {dom.group}, got {cod.group}.")
    return max((
        (function(act(g, v)) - act(h, function(v))).abs().max().item()
        for v in tensors for g, h in zip(dom.images, cod.images)),
        default=0.0)
