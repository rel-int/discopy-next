# -*- coding: utf-8 -*-

"""
Recursive message passing aggregated over the blocks of a group action
rather than gathered along a table.

A wiring invariant under a group of symmetries is a union of its orbitals,
and for a group built from symmetric groups by products and wreath
products every orbital is a signed sum of *blocks*, the points agreeing
with a given one on some axes of the tensor the group acts on, see
:func:`~discopy.neural.wiring.decompose`. Summing over a block is a
reduction over the axes it does not keep, so a round needs no gather and no
scatter, the loops a compiler fuses and vectorises: the geometric
counterpart of a kernel written by hand for one wiring.

:class:`Geometric` is the mixin that runs a
:class:`~discopy.neural.recursion.Recursion` this way. Its :meth:`cycle
<Geometric.cycle>` computes the same function as the reference one, from
and to the same state, rounded differently: every linear map between two
nonlinearities is composed where that makes a round cheaper (the second
layer of the message encoder into the gates) and kept apart where it does
not (a narrow belief between two wide maps), the maps reading the state
are one product and so are the maps reading the beliefs, and the clue and
answer gates are computed once per cycle. Everything else -- the detached
prefix, the heads, the halting, the trainers -- is the network's own.

Pass ``kernel="triton"`` to a geometric network to fuse its peer reductions
on CUDA float32 tensors, including their backward pass. This uses the same
signed blocks for every geometry, without an edge activation tensor or
atomic gradient accumulation. Strided slices of the folded maps are read
directly. Other devices and dtypes use the PyTorch path; factor geometries
are unchanged. Blocks larger than 1024 points and layouts requiring
64-bit indexing also use PyTorch.
The default ``kernel="torch"`` supports higher derivatives,
whereas the fused path supports first derivatives only. Triton is needed
only for the fused path, and no global precision flags are changed.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Shift
    Port
    Peers
    Factors
    Maps
    Geometric

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        view
        unview
        pool
        spread
        pairs
        linear
        fold

Example
-------
A four by four sudoku: its symmetries permute the rows within a band, the
bands, and the same for the columns, so its peer graph is three families of
blocks less their overlaps and the geometric network computes what the
peer network does.

>>> import torch  # doctest: +EXTRA
>>> from discopy.group import Product, Symmetric, Wreath
>>> from discopy.neural.recursion import PeerNet
>>> from discopy.neural.wiring import Block, primal
>>> lines = Wreath[Symmetric[2], Symmetric[2]]
>>> units = sum((Block((2, 2, 2, 2), keep).members()
...              for keep in ((0, 1), (2, 3), (0, 2))), ())
>>> class GeometricPeerNet(Geometric, PeerNet):
...     group = Product[lines, lines]
>>> torch.manual_seed(0)  # doctest: +ELLIPSIS
<...>
>>> reference = PeerNet((4, 6, 8, 5), primal(units), 4, 2, 3, 1).double()
>>> geometric = GeometricPeerNet((4, 6, 8, 5), primal(units), 4, 2, 3, 1)
>>> _ = geometric.double().load_state_dict(reference.state_dict())
>>> x = torch.rand(2, 16, 4, dtype=torch.float64)
>>> s, t = reference.initial(x), geometric.initial(x)
>>> for _ in range(3):
...     s, t = reference.cycle(s), geometric.cycle(t)
>>> assert all(torch.allclose(a, b) for a, b in zip(s, t))
"""

from __future__ import annotations

from math import prod
from typing import NamedTuple

import torch
from torch.nn import functional as F

from discopy.neural.cell import gru
from discopy.neural.wiring import Block, chains, decompose, families


def view(block: Block, x: torch.Tensor) -> torch.Tensor:
    """
    A tensor on the points ``(B, N, h)`` as ``(B, count, size, h)``, one
    block per row.

    Parameters:
        block : The partition into blocks.
        x : The tensor, its points numbered row-major in ``block.shape``.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> columns = Block((2, 2), (1, ))
    >>> view(columns, torch.arange(4.).view(1, 4, 1)).flatten().tolist()
    [0.0, 2.0, 1.0, 3.0]
    """
    rows, width = x.shape[0], x.shape[-1]
    axes = (0, *(a + 1 for a in block.order), len(block.shape) + 1)
    return x.reshape(rows, *block.shape, width).permute(axes).reshape(
        rows, block.count, block.size, width)


def unview(block: Block, y: torch.Tensor) -> torch.Tensor:
    """
    The inverse of :func:`view`, from ``(B, count, size, h)`` to the points.

    Parameters:
        block : The partition into blocks.
        y : The tensor, one block per row.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> columns, x = Block((2, 2), (1, )), torch.arange(4.).view(1, 4, 1)
    >>> torch.equal(unview(columns, view(columns, x)), x)
    True
    """
    rows, width = y.shape[0], y.shape[-1]
    inverse = sorted(range(len(block.order)), key=block.order.__getitem__)
    axes = (0, *(i + 1 for i in inverse), len(block.shape) + 1)
    return y.reshape(rows, *(block.shape[a] for a in block.order),
                     width).permute(axes).reshape(rows, -1, width)


def pool(block: Block, x: torch.Tensor, mean: bool = False) -> torch.Tensor:
    """
    The sum, or the mean, of a tensor over every block, ``(B, count, h)``.

    Parameters:
        block : The partition into blocks.
        x : The tensor on the points, ``(B, N, h)``.
        mean : Whether to take the mean rather than the sum.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> rows = Block((2, 3), (0, ))
    >>> pool(rows, torch.arange(6.).view(1, 6, 1)).flatten().tolist()
    [3.0, 12.0]
    """
    blocks = view(block, x)
    return blocks.mean(2) if mean else blocks.sum(2)


def spread(block: Block, y: torch.Tensor) -> torch.Tensor:
    """
    Every block's value on every one of its points, ``(B, N, h)``.

    Parameters:
        block : The partition into blocks.
        y : One value per block, ``(B, count, h)``.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> columns = Block((2, 2), (1, ))
    >>> spread(columns, torch.tensor([[[0.], [1.]]])).flatten().tolist()
    [0.0, 1.0, 0.0, 1.0]
    """
    rows, width = y.shape[0], y.shape[-1]
    return unview(block, y.unsqueeze(2).expand(
        rows, block.count, block.size, width))


def pairs(block: Block, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    ``relu(s[v] + x[w])`` summed over the points ``w`` of the block of every
    point ``v``, by broadcasting each block against itself.

    Parameters:
        block : The partition into blocks.
        s : The receivers' half, ``(B, N, h)``.
        x : The senders' half, ``(B, N, h)``.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> rows = Block((2, 2), (0, ))
    >>> s, x = torch.zeros(1, 4, 1), torch.tensor([[[1.], [-1.], [2.], [3.]]])
    >>> pairs(rows, s, x).flatten().tolist()
    [1.0, 1.0, 5.0, 5.0]
    """
    if not block.free:
        return torch.relu(s + x)
    return unview(block, torch.relu(
        view(block, s).unsqueeze(3) + view(block, x).unsqueeze(2)).sum(3))


class Shift(NamedTuple):
    """
    A port reading the point ``step`` away along ``axis`` of a grid of shape
    ``shape``, zero off the grid: the translations of the grid, cut at its
    edges.

    Parameters:
        shape : The shape of the grid.
        axis : The axis of the step.
        step : The step, negative for the preceding points.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> up = Shift((2, 2), 0, -1)
    >>> up(torch.arange(1., 5.).view(1, 4, 1)).flatten().tolist()
    [0.0, 0.0, 1.0, 2.0]
    """
    shape: tuple[int, ...]
    axis: int
    step: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows, width = x.shape[0], x.shape[-1]
        n, step = self.shape[self.axis], self.step
        moved = x.reshape(rows, *self.shape, width).narrow(
            self.axis + 1, max(step, 0), n - abs(step))
        padding = [0, 0] * (len(self.shape) - self.axis) + [
            max(-step, 0), max(step, 0)]
        return F.pad(moved, padding).reshape(rows, -1, width)


class Port(NamedTuple):
    """
    A factor port: the reply of the factor module ``module`` to the beliefs
    of every block of ``block``, pooled by a sum or a ``mean``.

    Parameters:
        module : The attribute name of the factor module.
        block : The partition of the points into factors.
        mean : Whether the factor pools by a mean rather than a sum.

    Example
    -------
    >>> Port("line", Block((3, 3), (0, )), mean=True).block.count
    3
    """
    module: str
    block: Block
    mean: bool = False


class Peers(NamedTuple):
    """
    A peer wiring as a signed sum of blocks, see
    :func:`~discopy.neural.wiring.decompose`, and the degree it divides by.

    Parameters:
        terms : The ``(coefficient, block)`` pairs.
        degree : The number of peers of every point.

    Example
    -------
    The four others on a square's two lines through a point:

    >>> from discopy.group import Product, Symmetric
    >>> square = Product[Symmetric[3], Symmetric[3]]
    >>> lines = ((0, 1), (2, 3))
    >>> peers = [[w for w in range(9) if w != v and any(
    ...     v // 3 == w // 3 if a else v % 3 == w % 3 for a in (0, 1))]
    ...     for v in range(9)]
    >>> wiring = Peers.of(peers, square)
    >>> wiring.degree, [(c, b.keep) for c, b in wiring.terms]
    (4, [(1, (1,)), (1, (0,)), (-2, (0, 1))])
    """
    terms: tuple
    degree: int

    def __call__(self, s: torch.Tensor, x: torch.Tensor,
                 kernel: str = "torch") -> torch.Tensor:
        """
        The mean of ``relu(s[v] + x[w])`` over the signed block relation.

        ``kernel="triton"`` fuses all terms and their first derivatives on
        CUDA float32 tensors with blocks of at most 1024 points, fewer
        than ``2**31`` elements and input element offsets below ``2**31``
        relative to each data pointer. Larger blocks or layouts, other
        devices and other dtypes use PyTorch; the
        default ``"torch"`` also supports higher derivatives. Triton is
        required only for the fused path. No precision settings change.

        Parameters:
            s : The receivers' half, ``(B, N, h)``.
            x : The senders' half, with the same shape, dtype and device.
            kernel : ``"torch"`` or ``"triton"``.
        """
        if kernel not in ("torch", "triton"):
            raise ValueError(f"unknown peer kernel: {kernel!r}")
        if not self.terms or self.degree <= 0:
            raise ValueError("peer reduction needs terms and positive degree")
        if s.ndim != 3 or s.shape != x.shape or s.dtype != x.dtype \
                or s.device != x.device or any(
                    prod(block.shape) != s.shape[1]
                    for _, block in self.terms):
            raise ValueError("peer inputs need matching (B, N, h) layouts")
        if not s.numel():
            return s + x
        if kernel == "triton" and s.is_cuda and s.dtype == torch.float32 \
                and s.numel() < 2**31 \
                and all(block.size <= 1024 for _, block in self.terms) \
                and all(
                    sum((size - 1) * stride
                        for size, stride in zip(t.shape, t.stride())) < 2**31
                    for t in (s, x)):
            # pylint: disable=import-outside-toplevel  # CUDA-only dependency
            from discopy.neural.kernels import encode, peer_pairs
            return peer_pairs(s, x, encode(self.terms), self.degree)
        total = None
        for coefficient, block in self.terms:
            term = pairs(block, s, x)
            term = term if coefficient == 1 else coefficient * term
            total = term if total is None else total + term
        return total / self.degree

    @classmethod
    def of(cls, relation, group) -> Peers:
        """
        The peer wiring of a relation invariant under a group.

        Parameters:
            relation : Per point, its peers, as many for every point.
            group : A group of symmetries of the relation, see
                    :func:`~discopy.neural.wiring.chains`.
        """
        degrees = {len(peers) for peers in relation}
        if len(degrees) != 1:
            raise ValueError(f"every point needs as many peers: {degrees}")
        return cls(decompose(relation, chains(group)), degrees.pop())


class Factors(NamedTuple):
    """
    A factor wiring: the ports of every point, in order, each a
    :class:`Shift` or a :class:`Port`.

    Parameters:
        ports : The ports.

    Example
    -------
    The rows and the columns of a square, its factors:

    >>> from discopy.group import Product, Symmetric
    >>> table = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
    ...          (0, 3, 6), (1, 4, 7), (2, 5, 8))
    >>> wiring = Factors.of(table, Product[Symmetric[3], Symmetric[3]])
    >>> [(p.module, p.block.keep) for p in wiring.ports]
    [('unit', (0,)), ('unit', (1,))]
    >>> wiring.modules
    ('unit',)
    """
    ports: tuple

    @classmethod
    def of(cls, factors, group, module: str = "unit", mean: bool = False,
           shifts: tuple = ()) -> Factors:
        """
        The factor wiring of a factor table whose factors are blocks of a
        group action, after the ``shifts``.

        Parameters:
            factors : The factor table, see
                      :func:`~discopy.neural.wiring.families`.
            group : The group, see :func:`~discopy.neural.wiring.chains`.
            module : The attribute name of the factor module.
            mean : Whether the factors pool by a mean.
            shifts : The direct ports, before the factor ports.
        """
        return cls(tuple(shifts) + tuple(
            Port(module, block, mean)
            for block in families(factors, chains(group))))

    @property
    def modules(self) -> tuple[str, ...]:
        """ The factor modules, in the order the ports first use them. """
        return tuple(dict.fromkeys(
            p.module for p in self.ports if isinstance(p, Port)))


def linear(x: torch.Tensor, weight: torch.Tensor,
           bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    ``F.linear`` as one matrix product over every leading axis: the
    batched product a strided input turns into is several times slower when
    its inner axis is narrow.

    Parameters:
        x : The input.
        weight : The matrix.
        bias : The bias.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> x = torch.ones(2, 3, 4).transpose(0, 1)
    >>> linear(x, torch.ones(5, 4)).shape
    torch.Size([3, 2, 5])
    """
    flat = F.linear(x.reshape(-1, x.shape[-1]), weight, bias)
    return flat.reshape(*x.shape[:-1], flat.shape[-1])


class Maps(NamedTuple):
    """
    The maps of a network as a round applies them, see :func:`fold`.

    Parameters:
        state : The stacked maps reading the state, ``(weight, bias)``: the
            state half of the message encoder, the hidden gates and the
            belief the state emits.
        state_sizes : Their output widths.
        beliefs : The stacked maps reading the beliefs: the belief half of
            the message encoder, then each factor module's embedding and
            member half.
        beliefs_sizes : Their output widths.
        port : The belief half of the message encoder.
        gate : The second layer of the encoder composed into the input
            gates, and its bias.
        clue : The clue's input gates.
        answer : The answer's input gates.
        norm : The weight, bias and epsilon of the cell's norm.
        refresh_ih : The refresh's input gates.
        refresh_hh : The refresh's hidden gates.
        refresh_norm : The refresh's norm.
        factors : Per factor module, its name, its pooled half and its
            output layer.
    """
    state: tuple
    state_sizes: tuple
    beliefs: tuple
    beliefs_sizes: tuple
    port: torch.Tensor
    gate: tuple
    clue: torch.Tensor
    answer: torch.Tensor
    norm: tuple
    refresh_ih: tuple
    refresh_hh: tuple
    refresh_norm: tuple
    factors: tuple


def fold(net, modules: tuple[str, ...] = ()) -> Maps:
    """
    The maps of a network's cell, refresh and factor modules, composed where
    composing makes a round cheaper: the second layer of the message
    encoder into the GRU's input gates, two wide maps, but not the belief
    into the port maps, a narrow belief between two wide maps. The maps
    reading the state are stacked into one product, and so are the maps
    reading the beliefs.

    Composition runs inside the autograd graph: the gradient of a loss of
    a geometric cycle reaches the parameters of the network.

    Parameters:
        net : The network, a :class:`~discopy.neural.recursion.Recursion`.
        modules : The attribute names of its factor modules.

    Example
    -------
    >>> from discopy.neural.recursion import PeerNet  # doctest: +EXTRA
    >>> maps = fold(PeerNet((4, 6, 8, 5), ((1, ), (0, )), 3, 2, 3, 1))
    >>> maps.state_sizes, maps.beliefs_sizes
    ((8, 18, 4), (8,))
    """
    # pylint: disable=too-many-locals  # one name per map of the round
    cell, refresh = net.cell, net.refresh
    first, second = cell.encode[0], cell.encode[2]
    sd, dim = cell.update.hidden_size, cell.emit.out_features
    hidden = first.out_features
    w_ih = cell.update.weight_ih
    on_state = [(first.weight[:, :sd], first.bias),
                (cell.update.weight_hh, cell.update.bias_hh),
                (cell.emit.weight, cell.emit.bias)]
    on_beliefs = [(first.weight[:, sd:], torch.zeros_like(first.bias))]
    factors = []
    for name in modules:
        unit = getattr(net, name)
        phi, rho = unit.phi[0], unit.rho[0]
        on_beliefs.append((torch.cat([phi.weight, rho.weight[:, :dim]]),
                           torch.cat([phi.bias, rho.bias])))
        factors.append((name, rho.weight[:, dim:], unit.rho[2].weight,
                        unit.rho[2].bias))
    return Maps(
        (torch.cat([w for w, _ in on_state]),
         torch.cat([b for _, b in on_state])),
        tuple(len(b) for _, b in on_state),
        (torch.cat([w for w, _ in on_beliefs]),
         torch.cat([b for _, b in on_beliefs])),
        tuple(len(b) for _, b in on_beliefs),
        first.weight[:, sd:],
        (w_ih[:, :hidden] @ second.weight,
         w_ih[:, :hidden] @ second.bias + cell.update.bias_ih),
        w_ih[:, hidden:hidden + dim], w_ih[:, hidden + dim:],
        (cell.norm.weight, cell.norm.bias, cell.norm.eps),
        (refresh.module.weight_ih, refresh.module.bias_ih),
        (refresh.module.weight_hh, refresh.module.bias_hh),
        (refresh.norm.weight, refresh.norm.bias, refresh.norm.eps),
        tuple(factors))


class Geometric:
    """
    The mixin running a :class:`~discopy.neural.recursion.Recursion` along
    the blocks of its geometry: put it before the network in the bases and
    give the subclass a :attr:`group`, from which the geometry is read off
    the network's own table, or a :attr:`geometry`, taken as it is.

    The mixin overrides :meth:`cycle`, a function of the same state to the
    same state, and sets :attr:`split`: the network keeps its parameters,
    their names, its checkpoints and its trainers. Its other methods are
    the pieces of a geometric cycle.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Product, Symmetric
    >>> from discopy.neural.recursion import FactorNet
    >>> table = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
    ...          (0, 3, 6), (1, 4, 7), (2, 5, 8))
    >>> class Square(Geometric, FactorNet):
    ...     group = Product[Symmetric[3], Symmetric[3]]
    >>> net = Square((4, 6, 8, 5), table, 2, 2, 3, 1)
    >>> [p.block.keep for p in net.geometry.ports]
    [(0,), (1,)]
    >>> class Wrong(Geometric, FactorNet):
    ...     pass
    >>> Wrong((4, 6, 8, 5), table, 2, 2, 3, 1)
    Traceback (most recent call last):
        ...
    TypeError: Wrong needs a group or a geometry.
    """
    group: type | None = None
    """ A group of symmetries of the network's table, see
    :func:`~discopy.neural.wiring.chains`. """

    geometry: Peers | Factors | None = None
    """ The wiring as blocks, read off the table when :attr:`group` is
    given, else given as it is and not checked against the network. """

    rounds: int
    """ The rounds of a cycle, the network's. """

    split: bool = True
    """ The layout of a factor network: the beliefs as emitted. """

    peer_kernel: str = "torch"
    """ Peer reduction implementation: ``"torch"`` or ``"triton"``.
    Selected by the constructor's ``kernel`` keyword. Factor geometries keep
    their PyTorch implementation. See :meth:`Peers.__call__`. """

    def __init__(self, *args, kernel: str = "torch", **kwargs):
        if kernel not in ("torch", "triton"):
            raise ValueError(f"unknown peer kernel: {kernel!r}")
        super().__init__(*args, **kwargs)
        self.peer_kernel = kernel
        if self.group is not None:
            self.geometry = self.derive()
        elif self.geometry is None:
            raise TypeError(
                f"{type(self).__name__} needs a group or a geometry.")

    def derive(self) -> Peers | Factors:
        """
        The geometry of the network's own table under :attr:`group`: its
        peers, or its factor table, every factor a block.

        Raises:
            ValueError : If the table is padded, or not a union of blocks.
        """
        # pylint: disable=no-member  # the network's, the mixin is mixed in
        peers = getattr(self, "peers", None)
        if peers is not None:
            return Peers.of(peers.tolist(), self.group)
        if getattr(self, "port_valid", None) is not None:
            raise ValueError("a padded factor table has no blocks")
        table = self.to_units.view(self.n_units, self.arity).tolist()
        return Factors.of(table, self.group, self.factor)

    def maps(self) -> Maps:
        """ The network's maps, see :func:`fold`. """
        modules = self.geometry.modules \
            if isinstance(self.geometry, Factors) else ()
        return fold(self, modules)

    def on_state(self, maps: Maps, state: torch.Tensor) -> tuple:
        """
        The maps reading the state, as one product: ``(shared, gh,
        emitted)``, the state half of the message encoder, the hidden gates
        and the beliefs the state emits.

        Parameters:
            maps : The maps.
            state : The state, ``(B, N, state_dim)``.
        """
        return tuple(linear(state, *maps.state).split(maps.state_sizes, -1))

    def emit(self, maps: Maps, state: torch.Tensor) -> torch.Tensor:
        """
        The beliefs a state emits.

        Parameters:
            maps : The maps.
            state : The state.
        """
        dim = maps.state_sizes[-1]
        return linear(state, maps.state[0][-dim:], maps.state[1][-dim:])

    def read(self, maps: Maps, beliefs: torch.Tensor) -> tuple:
        """
        The port map and the factor modules' first maps on the beliefs:
        ``(per_cell, parts)``.

        Parameters:
            maps : The maps.
            beliefs : The beliefs, ``(B, N, dim)``.
        """
        per_cell, *wide = linear(beliefs, *maps.beliefs).split(
            maps.beliefs_sizes, -1)
        parts = {}
        for (name, *_), both in zip(maps.factors, wide):
            half = both.shape[-1] // 2
            parts[name] = (torch.relu(both[..., :half]), both[..., half:])
        return per_cell, parts

    def update(self, maps: Maps, state, pooled, const, gh) -> torch.Tensor:
        """
        The gates, the GRU and the norm of a round.

        Parameters:
            maps : The maps.
            state : The state the round starts in.
            pooled : The pooled messages.
            const : The clue and answer gates of the cycle.
            gh : The hidden gates.
        """
        gi = linear(pooled, maps.gate[0]) + const
        return F.layer_norm(gru(gi, gh, state), state.shape[-1:],
                            *maps.norm)

    def close(self, maps: Maps, state, answer) -> torch.Tensor:
        """
        The answer refresh closing a cycle.

        Parameters:
            maps : The maps.
            state : The state after the last round.
            answer : The answer trace.
        """
        gi = linear(state, *maps.refresh_ih)
        gh = linear(answer, *maps.refresh_hh)
        return F.layer_norm(gru(gi, gh, answer), answer.shape[-1:],
                            *maps.refresh_norm)

    def factor_replies(self, maps: Maps, parts: dict) -> list:
        """
        The reply of every factor port, as a belief: the ports of a module
        share its last map, run once on all of them.

        Parameters:
            maps : The maps.
            parts : The embedding and the member half of every module.
        """
        hidden, order = {}, []
        weights = {name: rest for name, *rest in maps.factors}
        for port in self.geometry.ports:
            if isinstance(port, Port):
                embedded, member = parts[port.module]
                pooled = linear(pool(port.block, embedded, port.mean),
                                weights[port.module][0])
                found = hidden.setdefault(port.module, [])
                found.append(torch.relu(member + spread(port.block, pooled)))
                order.append((port.module, len(found) - 1))
        out = {name: linear(torch.stack(found, 2),
                            *weights[name][1:]).unbind(2)
               for name, found in hidden.items()}
        return [out[name][i] for name, i in order]

    def cycle(self, s):
        """
        :attr:`rounds` rounds and the refresh, on the blocks of the
        geometry: the same state as the network's own cycle, rounded
        differently.

        Parameters:
            s : The carried state.
        """
        maps = self.maps()
        const = linear(s.clue, maps.clue) + linear(
            s.answer, maps.answer, maps.gate[1])
        if isinstance(self.geometry, Peers):
            state, units = self.peer_rounds(maps, s, const)
            return s._replace(units=units, state=state,
                              answer=self.close(maps, state, s.answer))
        state, units, cells = self.factor_rounds(maps, s, const)
        return s._replace(cells=cells, units=units, state=state,
                          answer=self.close(maps, state, s.answer))

    def peer_rounds(self, maps: Maps, s, const) -> tuple:
        """
        The rounds of a peer cycle: ``(state, units)`` after the last. A
        round reads the beliefs emitted from the state it starts in, the
        first one those the cycle carries.

        Parameters:
            maps : The maps.
            s : The carried state.
            const : The clue and answer gates of the cycle.
        """
        assert isinstance(self.geometry, Peers)
        state, carried = s.state, s.units
        for _ in range(self.rounds):
            shared, gh, emitted = self.on_state(maps, state)
            per_cell, _ = self.read(
                maps, emitted if carried is None else carried)
            pooled = self.geometry(shared, per_cell, kernel=self.peer_kernel)
            state = self.update(maps, state, pooled,
                                const, gh)
            carried = None
        return state, self.emit(maps, state)

    def factor_rounds(self, maps: Maps, s, const) -> tuple:
        """
        The rounds of a factor cycle: ``(state, units, cells)`` after the
        last, the beliefs and the messages on the ports. A round reads the
        beliefs emitted from the state it starts in, and the factors reply
        to them for the round after; the first round reads the messages and
        the beliefs the cycle carries.

        Parameters:
            maps : The maps.
            s : The carried state.
            const : The clue and answer gates of the cycle.
        """
        # pylint: disable=too-many-locals  # one name per tensor of a round
        state, carried = s.state, s.units
        messages = linear(s.cells, maps.port).unbind(2)
        replied: list = []
        for _ in range(self.rounds):
            shared, gh, emitted = self.on_state(maps, state)
            per_cell, parts = self.read(
                maps, emitted if carried is None else carried)
            if carried is None:
                messages = self.messages(maps, per_cell, replied)
            total = None
            for message in messages:
                term = torch.relu(shared + message)
                total = term if total is None else total + term
            replied = self.factor_replies(maps, parts)
            state = self.update(maps, state, total / len(messages), const, gh)
            carried = None
        units = self.emit(maps, state)
        return state, units, self.carry(units, replied)

    def messages(self, maps: Maps, per_cell, replied) -> list:
        """
        The messages on the ports, in the Site's hidden space: the shifted
        beliefs and the factor replies of the round before.

        Parameters:
            maps : The maps.
            per_cell : The port map of the beliefs.
            replied : The replies of the round before, as beliefs.
        """
        encoded = iter(linear(torch.stack(replied, 2), maps.port).unbind(2)
                       if replied else ())
        return [next(encoded) if isinstance(port, Port) else port(per_cell)
                for port in self.geometry.ports]

    def carry(self, units, replied) -> torch.Tensor:
        """
        The messages a factor cycle hands the next, as beliefs.

        Parameters:
            units : The beliefs after the last round.
            replied : The replies of the last round.
        """
        found = iter(replied)
        return torch.stack([next(found) if isinstance(port, Port)
                            else port(units)
                            for port in self.geometry.ports], 2)
