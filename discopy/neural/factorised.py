# -*- coding: utf-8 -*-

"""
Equivariant layers factorised along the construction of the group: the
realisations of :mod:`discopy.neural.equivariant` that compute the linear maps
from the sums over the blocks of a product, a wreath product or a power and
the correlations of a cyclic group, instead of from a dense matrix with one
entry for each pair of points.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Pooling
    Kronecker
    Factor
    Structure

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        realise
        structure
        canonical
        nested
        block_shape

Example
-------
A realisation is chosen by the structure of the actions alone, and it is
equal to the dense layer, here the maps of the symmetric group on three
points, one weight on the diagonal and one off it:

>>> import torch  # doctest: +EXTRA
>>> from discopy.neural.equivariant import Dense
>>> action = Symmetric[3].action()
>>> fast = realise(action, action)
>>> type(fast).__name__, fast.orbits
('Kronecker', 2)
>>> v, w = torch.tensor([[1.], [2.], [4.]]), torch.tensor([[[2.]], [[1.]]])
>>> dense = Dense(action, action)
>>> fast(v, w).flatten().tolist(), dense(v, w).flatten().tolist()
([8.0, 9.0, 11.0], [8.0, 9.0, 11.0])
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import numpy
import torch
from torch.nn import functional

from discopy.group import (
    Action, Cyclic, FiniteGroup, Product, Symmetric, Wreath)
from discopy.neural.equivariant import Dense, Realisation, labels
from discopy.neural.wiring import Block, chains, orbitals
from discopy.python.finset import Permutation


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


class Factor(NamedTuple):
    """
    A factor of the points of an action: the points are a row-major tensor
    whose shape is the concatenation of the shapes of the factors.

    A *chain* ``(n_1, ..., n_L)`` is the natural action of a symmetric group,
    of the cyclic group of order two or of an imprimitive wreath product of
    these, see :func:`discopy.neural.wiring.chains`: two points are in one
    of ``L + 1`` relations, how many of their first coordinates agree. A
    cyclic factor of order ``n`` has ``n`` relations, the difference of the
    coordinates modulo ``n``.

    Parameters:
        shape : The sizes of the axes of the factor.
        cyclic : Whether the factor is a cyclic group acting on itself.

    Example
    -------
    >>> Factor((3, 3)).orbits, Factor((5, ), cyclic=True).orbits
    (3, 5)
    """
    shape: tuple[int, ...]
    cyclic: bool = False

    @property
    def orbits(self) -> int:
        """
        The number of orbits on pairs of points of the factor, i.e. of ranks
        of a pair: the rank of a pair of a chain is its length less the
        number of coordinates they agree on.
        """
        return self.shape[0] if self.cyclic else len(self.shape) + 1


def nested(array: numpy.ndarray) -> tuple:
    """
    An array as nested tuples of integers.

    Parameters:
        array : The array.

    Example
    -------
    >>> nested(numpy.arange(4).reshape(2, 2))
    ((0, 1), (2, 3))
    """
    return tuple(map(nested, array)) if array.ndim > 1\
        else tuple(map(int, array))


class Structure(NamedTuple):
    """
    How the orbits on pairs of points of an action are laid out along its
    factors: the orbit of a pair is ``tie[r_1, ..., r_m]`` where ``r_i`` is
    the rank of the pair in the ``i``-th factor, see :class:`Factor`.

    Parameters:
        factors : The factors of the points.
        tie : The orbit of each tuple of ranks, as nested tuples.

    Example
    -------
    The orbits of a product are the pairs of orbits of its factors, numbered
    row-major, and those of a power action of a wreath product the orbits
    of the tuples of orbits under the top group:

    >>> chain = Structure.chain((3, ))
    >>> chain.tie, chain.product(chain).tie
    ((0, 1), ((0, 1), (2, 3)))
    >>> chain.power(Symmetric[2]).tie
    ((0, 1), (1, 2))
    """
    factors: tuple[Factor, ...]
    tie: tuple

    @property
    def table(self) -> numpy.ndarray:
        """ The tie as an array, of shape the ranks of the factors. """
        return numpy.array(self.tie)

    @property
    def shape(self) -> tuple[int, ...]:
        """ The shape of the points, the shapes of the factors in a row. """
        return sum((factor.shape for factor in self.factors), ())

    @classmethod
    def of(cls, factors: tuple[Factor, ...], table) -> Structure:
        """
        The structure of the given factors and tie array.

        Parameters:
            factors : The factors of the points.
            table : An array of shape the ranks of the factors.
        """
        return cls(tuple(factors), nested(numpy.asarray(table)))

    @classmethod
    def chain(cls, shape: tuple[int, ...]) -> Structure:
        """
        The structure of a chain: its orbits are its ranks.

        Parameters:
            shape : The sizes of the axes of the chain.
        """
        factor = Factor(tuple(shape))
        return cls.of((factor, ), range(factor.orbits))

    def product(self, other: Structure) -> Structure:
        """
        The structure of a product of groups: one factor after the other.

        Parameters:
            other : The structure of the right factor.
        """
        left, right = self.table, other.table
        table = left.reshape(left.shape + right.ndim * (1, ))\
            * (right.max() + 1) + right
        return self.of(self.factors + other.factors, table)

    def power(self, top: type[FiniteGroup]) -> Structure:
        """
        The structure of the power action of a wreath product by ``top`` of
        a group with this structure: ``k`` copies of the factors, whose
        tuples of orbits ``top`` permutes.

        Parameters:
            top : The group permuting the ``k`` copies.
        """
        table, k = self.table, top.action().size
        o, m = int(table.max()) + 1, table.ndim
        points = numpy.indices(k * (o, )).reshape(k, -1)

        def image(permutation: Permutation) -> Permutation:
            moved = numpy.empty_like(points)
            moved[list(permutation)] = points
            index = numpy.ravel_multi_index(tuple(moved), k * (o, ))
            return Permutation([int(i) for i in index], o ** k)

        action = Action(top, o ** k, tuple(map(image, top.action().images)))
        classes = labels(action).reshape(k * (o, ))
        indices = tuple(
            table.reshape((m * j) * (1, ) + table.shape
                          + (m * (k - 1 - j)) * (1, )) for j in range(k))
        return self.of(k * self.factors, classes[indices])


def block_shape(group: type) -> tuple[int, ...] | None:
    """
    The chain of the imprimitive action of a wreath product, if its axes are
    all of size at least two, see :func:`discopy.neural.wiring.chains`.

    Parameters:
        group : A wreath product.

    Example
    -------
    >>> block_shape(Wreath[Symmetric[3], Cyclic[2]])
    (2, 3)
    >>> block_shape(Wreath[Cyclic[3], Cyclic[2]]) is None
    True
    """
    try:
        shape = chains(group)[0]
    except TypeError:
        return None
    return shape if min(shape) >= 2 else None


@functools.cache
def canonical(group: type) -> tuple[tuple[int, Callable], ...]:
    """
    The actions of a group that have a :class:`Structure`, each as its number
    of points and a thunk of the action with its structure, so that only an
    action of the right size is ever built: the action of a symmetric group,
    of the cyclic group of order two and of an imprimitive wreath product of
    these, the action of a cyclic group of order at least three on itself,
    the power action of a wreath product and the products of such actions.

    Parameters:
        group : The group.

    Example
    -------
    >>> [len(canonical(group)) for group in
    ...  (Wreath[Symmetric[3], Cyclic[2]], Cyclic[4], Cyclic[1])]
    [2, 1, 0]
    >>> [size for size, _ in canonical(Wreath[Symmetric[3], Cyclic[2]])]
    [6, 9]
    """
    if issubclass(group, Product):
        def joined(left, right):
            a, b = left(), right()
            return group.action(a[0], b[0]), a[1].product(b[1])

        def pair(left, right):
            return left[0] * right[0], lambda: joined(left[1], right[1])
        return tuple(pair(a, b) for a in canonical(group.left)
                     for b in canonical(group.right))
    if issubclass(group, Wreath):
        shape, base = block_shape(group), structure(group.base.action())
        blocks = () if shape is None else ((math.prod(shape), lambda: (
            group.action(), Structure.chain(shape))), )
        tuples = () if base is None else (
            (group.base.action().size ** group.top.action().size, lambda: (
                group.action(power=True), base.power(group.top))), )
        return blocks + tuples
    if issubclass(group, Symmetric) and group.n >= 2\
            or issubclass(group, Cyclic) and group.n == 2:
        return ((group.n, lambda: (
            group.action(), Structure.chain((group.n, )))), )
    if issubclass(group, Cyclic) and group.n >= 3:
        factor = Factor((group.n, ), cyclic=True)
        return ((group.n, lambda: (
            group.action(), Structure.of((factor, ), range(group.n)))), )
    return ()


@functools.cache
def structure(action: Action) -> Structure | None:
    """
    The :class:`Structure` of an action, if it is the natural action of its
    group on a product of chains and cyclic groups, or a power of them: an
    action that is only equal to such an action up to a relabelling of its
    points has none. Only the candidates with as many points as the action
    are built.

    Parameters:
        action : The action.

    Example
    -------
    >>> structure(Symmetric[3].action())
    Structure(factors=(Factor(shape=(3,), cyclic=False),), tie=(0, 1))
    >>> structure(Symmetric[1].action()) is None
    True
    """
    candidates = (build() for size, build in canonical(action.group)
                  if size == action.size)
    return next((found for known, found in candidates
                 if known == action), None)


class Kronecker(Realisation):
    """
    The equivariant linear maps of the natural action of a product of chains
    and cyclic groups on itself, computed along the axes of the points
    instead of on the matrix: the orbits on pairs are the tuples of depths
    of agreement along the chains, see :func:`discopy.neural.wiring.orbitals`,
    and of differences along the cyclic axes. A tuple of depths is the
    difference of blocks of points, see :func:`discopy.neural.wiring.Block`.
    The output is the sum, over the blocks, of the sum of the input over each
    block, mixed by one matrix ``(D, C)`` for each point and spread back: it
    costs ``O(|X|C + |X|D)`` for each of the ``prod(L + 1)`` blocks instead
    of ``O(|X|^2 CD)`` for :class:`Dense`. The cyclic axes are mixed by a
    cross-correlation, directly as a circular convolution of at most three
    axes, or by the fast Fourier transform: the transform is used when
    ``fft`` is true, or when it is ``None`` and the cyclic axes have at
    least :attr:`fft_from` points together or there are more than three.

    Parameters:
        dom : The action on the input points.
        cod : The action on the output points, equal to ``dom``.
        fft : Whether to use the transform, ``None`` to choose by size.

    Attributes:
        structure : The :class:`Structure` of the action.
        fft : Whether to use the transform, ``None`` to choose by size.
        fft_from : The number of points of the cyclic axes from which the
                   transform is used, if ``fft`` is ``None``.

    Raises:
        TypeError : If the action has no structure, or ``dom`` and ``cod``
                    are different actions, or ``fft`` is false with more
                    than three cyclic axes.

    Example
    -------
    The maps of a symmetric group have ``w[0]`` on the diagonal and
    ``w[1]`` off it, those of a cyclic group are circulant:

    >>> import torch  # doctest: +EXTRA
    >>> action = Symmetric[3].action()
    >>> kronecker = Kronecker(action, action)
    >>> kronecker.orbits, kronecker.tying().tolist()
    (2, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> v, w = torch.tensor([[1.], [2.], [4.]]), torch.tensor([[[2.]], [[1.]]])
    >>> kronecker(v, w).flatten().tolist()
    [8.0, 9.0, 11.0]
    >>> rotations = Cyclic[3].action()
    >>> shift = torch.tensor([[[0.]], [[1.]], [[0.]]])
    >>> Kronecker(rotations, rotations)(v, shift).flatten().tolist()
    [2.0, 4.0, 1.0]
    >>> fourier = Kronecker(rotations, rotations, fft=True)
    >>> fourier(v, shift).round().flatten().tolist()
    [2.0, 4.0, 1.0]
    """
    fft_from = 64

    def __init__(self, dom: Action, cod: Action, fft: bool | None = None):
        found = structure(dom) if dom == cod else None
        if found is None:
            raise TypeError("Expected the same action of a product of "
                            f"chains and cyclic groups twice, got {dom} "
                            f"and {cod}.")
        if fft is False and len(self.axes(found)) > 3:
            raise TypeError("Expected at most three cyclic axes without "
                            f"the Fourier transform, got {found.shape}.")
        table = found.table
        super().__init__(dom, cod, int(table.max()) + 1)
        self.structure, self.fft = found, fft
        self.register_buffer("tie", torch.from_numpy(table), persistent=False)

    @staticmethod
    def axes(found: Structure, offset: int = 0) -> tuple[int, ...]:
        """
        The axes of the points that are cyclic.

        Parameters:
            found : The structure of the action.
            offset : The number of axes before the points, e.g. the batch.
        """
        start, axes = offset, []
        for factor in found.factors:
            axes += [start] if factor.cyclic else []
            start += len(factor.shape)
        return tuple(axes)

    def tying(self) -> numpy.ndarray:
        """ The tie indexed by the rank of each pair along each factor. """
        coordinates = numpy.indices(self.structure.shape).reshape(
            len(self.structure.shape), -1)
        ranks, start = [], 0
        for factor in self.structure.factors:
            axes = coordinates[start:start + len(factor.shape)]
            agree = axes[:, :, None] == axes[:, None, :]
            ranks.append((axes[0][None, :] - axes[0][:, None])
                         % factor.shape[0] if factor.cyclic else
                         len(factor.shape) - numpy.cumprod(agree, 0).sum(0))
            start += len(factor.shape)
        tie = self.tie.cpu().numpy()  # ty: ignore[call-non-callable]
        return tie[tuple(ranks)]

    @staticmethod
    def pool(x: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        """
        The sum over some axes, kept as axes of size one.

        Parameters:
            x : The tensor.
            axes : The axes to sum over, none included.
        """
        return x.sum(axes, keepdim=True) if axes else x

    def forward(self, v: torch.Tensor, w: torch.Tensor,
                b: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply the layer, see :meth:`Dense.forward`.

        Parameters:
            v : The input, of shape ``(..., |X|, C)``.
            w : The weights, of shape ``(orbits, D, C)``.
            b : The biases, of shape ``(bias_orbits, D)``, if any.
        """
        factors = self.structure.factors
        batch = v.shape[:-2]
        x = v.reshape(*batch, *self.structure.shape, v.shape[-1])
        u = w[self.tie]  # ty: ignore[invalid-argument-type]
        for axis, factor in enumerate(factors):
            zero = torch.zeros_like(u.narrow(axis, 0, 1))
            u = u if factor.cyclic else u.flip(axis).diff(
                dim=axis, prepend=zero)
        sizes = tuple(f.shape for f in factors if not f.cyclic)
        out = sum(self.term(x, u, depth, len(batch))
                  for depth in orbitals(sizes))
        return self.bias(out.reshape(*batch, -1, out.shape[-1]), b)

    def term(self, x: torch.Tensor, u: torch.Tensor,
             depth: tuple[int, ...], offset: int) -> torch.Tensor:
        """
        The term of one tuple of depths: the sum of the input over the blocks
        of points agreeing to these depths along the chains, correlated
        with their weights along the cyclic axes.

        Parameters:
            x : The input, of shape ``(..., *shape, C)``.
            u : The weights by depth, of shape ``(*ranks, D, C)``.
            depth : One depth for each chain.
            offset : The number of axes of ``x`` before the points.
        """
        factors, depths = self.structure.factors, iter(depth)
        agreed = tuple(1 if f.cyclic else next(depths) for f in factors)
        index = tuple(slice(None) if f.cyclic else a
                      for f, a in zip(factors, agreed))
        layout = tuple(f.shape for f in factors)
        free = tuple(offset + a for a in Block.of(layout, agreed).free)
        axes = self.axes(self.structure, offset)
        return self.correlate(self.pool(x, free), u[index], axes)

    def correlate(self, p: torch.Tensor, k: torch.Tensor,
                  axes: tuple[int, ...]) -> torch.Tensor:
        """
        The mix of the channels with a cross-correlation along some cyclic
        axes: ``out[y] = sum(k[s] @ p[y + s])`` where ``y + s`` is modulo the
        size of each axis, the kernel is never flipped.

        Parameters:
            p : The input, of shape ``(..., C)``, the axes of size one
                broadcast.
            k : The kernel, of shape ``(*sizes, D, C)``.
            axes : The axes of ``p`` the kernel runs over, none included.
        """
        if not axes:
            return torch.einsum("...c,dc->...d", p, k)
        sizes, last = k.shape[:len(axes)], tuple(range(-len(axes) - 1, -1))
        fourier = self.fft if self.fft is not None else\
            math.prod(sizes) >= self.fft_from or len(axes) > 3
        method = self.fourier if fourier else self.direct
        return method(p.movedim(axes, last), k).movedim(last, axes)

    @staticmethod
    def fourier(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        The cross-correlation along the axes before the channels, by the
        Fourier transform: the conjugate of the transform of the kernel.

        Parameters:
            p : The input, of shape ``(..., *sizes, C)``.
            k : The kernel, of shape ``(*sizes, D, C)``.
        """
        # pylint: disable=not-callable  # torch.fft is compiled, uninferable
        n = k.ndim - 2
        sizes, dims = k.shape[:n], tuple(range(-n - 1, -1))
        spectrum = torch.einsum(
            "...c,...dc->...d", torch.fft.rfftn(p, dim=dims),
            torch.fft.rfftn(k, dim=tuple(range(n))).conj())
        return torch.fft.irfftn(spectrum, s=sizes, dim=dims)

    @staticmethod
    def direct(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        The cross-correlation along the axes before the channels, by a
        convolution of the input padded circularly, at most three axes.

        Parameters:
            p : The input, of shape ``(..., *sizes, C)``.
            k : The kernel, of shape ``(*sizes, D, C)``.
        """
        n = k.ndim - 2
        sizes, shape = k.shape[:n], p.shape[:-n - 1]
        x = p.reshape(-1, *sizes, p.shape[-1]).movedim(-1, 1)
        pad = tuple(m for size in reversed(sizes) for m in (0, size - 1))
        conv = (functional.conv1d, functional.conv2d, functional.conv3d)[n - 1]
        out = conv(functional.pad(x, pad, mode="circular"),
                   k.movedim((-2, -1), (0, 1)))
        return out.movedim(1, -1).reshape(*shape, *sizes, k.shape[-2])


def realise(dom: Action, cod: Action) -> Realisation:
    """
    The fastest realisation of the equivariant linear maps from ``dom`` to
    ``cod``, chosen by the structure of the actions alone: :class:`Pooling`
    if a side has one point, else :class:`Kronecker` if they are the same
    action with a :func:`structure`, else :class:`Dense`. Whichever is
    chosen, it numbers its orbits as
    :func:`discopy.neural.equivariant.tying` and agrees with :class:`Dense`.

    Parameters:
        dom : The action on the input points.
        cod : The action on the output points.

    Example
    -------
    >>> action = Symmetric[3].action()
    >>> trivial = Action.trivial(Symmetric[3])
    >>> [type(realise(a, b)).__name__ for a, b in (
    ...     (action, trivial), (action, action),
    ...     (Cyclic[3].action(), Cyclic[3].action()))]
    ['Pooling', 'Kronecker', 'Kronecker']
    """
    if dom.size == 1 or cod.size == 1:
        return Pooling(dom, cod)
    if dom == cod and structure(dom) is not None:
        return Kronecker(dom, cod)
    return Dense(dom, cod)
