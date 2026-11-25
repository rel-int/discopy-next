# -*- coding: utf-8 -*-

"""
Equivariant layers factorised along the construction of the group: the
realisations of :mod:`discopy.neural.equivariant` that compute the linear maps
from the sums over the orbits of the points instead of from a dense matrix
with one entry for each pair of points.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Pooling

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        realise

Example
-------
A realisation is chosen by the structure of the actions alone, and it is
equal to the dense layer, here the invariant maps of the symmetric group on
three points, one weight on the orbit of all the points:

>>> import torch  # doctest: +EXTRA
>>> from discopy.group import Symmetric
>>> action = Symmetric[3].action()
>>> trivial = Action.trivial(Symmetric[3])
>>> fast = realise(action, trivial)
>>> type(fast).__name__, fast.orbits
('Pooling', 1)
>>> v, w = torch.tensor([[1.], [2.], [4.]]), torch.tensor([[[2.]]])
>>> dense = Dense(action, trivial)
>>> fast(v, w).flatten().tolist(), dense(v, w).flatten().tolist()
([14.0], [14.0])
"""

from __future__ import annotations

import numpy
import torch

from discopy.group import Action
from discopy.neural.equivariant import Dense, Realisation, labels


class Pooling(Realisation):
    """
    The equivariant linear maps with a single point on one side, i.e. the
    invariant maps and the broadcasts, of any group and any action: the
    orbits on pairs are those of the other side. An input is summed over the
    orbits of its points before the weights are applied, and an output is
    copied to the points of each orbit after, so it costs ``O(|X|C + OCD)``
    where :class:`Dense` costs ``O(|X|CD)`` with a gathered weight.

    Parameters:
        dom : The action on the input points.
        cod : The action on the output points.

    Raises:
        TypeError : If neither action is on a single point.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Symmetric
    >>> action = Symmetric[3].action()
    >>> pooling = Pooling(action, Action.trivial(Symmetric[3]))
    >>> pooling.orbits, pooling.tying().tolist()
    (1, [[0, 0, 0]])
    >>> v, w = torch.tensor([[1.], [2.], [4.]]), torch.tensor([[[1.]]])
    >>> pooling(v, w).flatten().tolist()
    [7.0]
    >>> Pooling(action, action)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    TypeError: Expected an action on a single point, got ...
    """
    def __init__(self, dom: Action, cod: Action):
        if dom.size != 1 and cod.size != 1:
            raise TypeError("Expected an action on a single point, "
                            f"got {dom.size} and {cod.size} points.")
        index = labels(dom if cod.size == 1 else cod)
        super().__init__(dom, cod, int(index.max()) + 1 if index.size else 0)
        self.into = cod.size == 1
        self.register_buffer(
            "index", torch.from_numpy(index), persistent=False)

    def tying(self) -> numpy.ndarray:
        """ The orbit of each point of the longer side, as a row or column. """
        index = self.index.cpu().numpy()  # ty: ignore[call-non-callable]
        return index.reshape(1, -1) if self.into else index.reshape(-1, 1)

    def forward(self, v: torch.Tensor, w: torch.Tensor,
                b: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply the layer, see :meth:`Dense.forward`.

        Parameters:
            v : The input, of shape ``(..., |X|, C)``.
            w : The weights, of shape ``(orbits, D, C)``.
            b : The biases, of shape ``(bias_orbits, D)``, if any.
        """
        if self.into:
            pooled = v.new_zeros(*v.shape[:-2], self.orbits, v.shape[-1])
            pooled = pooled.index_add(
                -2, self.index, v)  # ty: ignore[invalid-argument-type]
            out = torch.einsum("...oc,odc->...d", pooled, w).unsqueeze(-2)
            return self.bias(out, b)
        out = torch.einsum("...c,odc->...od", v[..., 0, :], w)
        out = out[..., self.index, :]  # ty: ignore[invalid-argument-type]
        return self.bias(out, b)


def realise(dom: Action, cod: Action) -> Realisation:
    """
    The fastest realisation of the equivariant linear maps from ``dom`` to
    ``cod``, chosen by the structure of the actions alone: :class:`Pooling`
    if a side has one point, else :class:`Dense`. Whichever is chosen, it
    numbers its orbits as :func:`discopy.neural.equivariant.tying` and agrees
    with :class:`Dense`.

    Parameters:
        dom : The action on the input points.
        cod : The action on the output points.

    Example
    -------
    >>> from discopy.group import Symmetric
    >>> action = Symmetric[3].action()
    >>> trivial = Action.trivial(Symmetric[3])
    >>> [type(realise(a, b)).__name__ for a, b in (
    ...     (action, trivial), (trivial, action), (action, action))]
    ['Pooling', 'Pooling', 'Dense']
    """
    if dom.size == 1 or cod.size == 1:
        return Pooling(dom, cod)
    return Dense(dom, cod)
