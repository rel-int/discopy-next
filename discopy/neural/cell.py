# -*- coding: utf-8 -*-

"""
The boxes of a recursive message-passing network, as PyTorch modules.

A *site* is the one module every node of a graph shares: it reads one
message per port, encodes each against its own state, pools the encodings,
updates its state with a GRU fed by the pool, the clue and the answer,
normalises it and emits one belief. A :class:`DeepSet` is the one module
every factor shares: it embeds its members' beliefs, sums the embeddings and
answers each member from its own belief and the sum. A :class:`Refresh`
updates the answer trace from the state once per cycle, and a
:class:`HaltHead` reads a halting logit off it. :mod:`discopy.neural.recursion`
wires them into a network.

Every module here computes exactly what it computed in the experiment it
comes from, operation for operation, so that a network built from them under
a seed has the same initial weights, the same parameter names and the same
outputs, bit for bit. Where two operation orders exist, both are kept and
named: :attr:`Site.split` chooses between them.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Widths
    Site
    PeerSite
    DeepSet
    Refresh
    HaltHead

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        as_widths
        count
        gru_cell
        gru
        widths_for

Example
-------
>>> import torch  # doctest: +EXTRA
>>> torch.manual_seed(0)  # doctest: +ELLIPSIS
<torch._C.Generator object at ...>
>>> w = Widths(dim=4, state_dim=6, hidden=8, y_dim=5)
>>> site, unit = Site(w), DeepSet(w)
>>> rows, cells, degree = 2, 3, 2
>>> messages = torch.zeros(rows, cells, degree, w.dim)
>>> state = torch.zeros(rows, cells, w.state_dim)
>>> clue, answer = torch.ones(rows, cells, w.dim), torch.ones(rows, cells, 5)
>>> belief, state = site(messages, state, clue, answer)
>>> belief.shape, state.shape
(torch.Size([2, 3, 4]), torch.Size([2, 3, 6]))
>>> unit(belief.reshape(rows, 1, cells, w.dim)).shape
torch.Size([2, 1, 3, 4])
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class Widths(NamedTuple):
    """
    The widths that fix the parameter count of a network.

    Parameters:
        dim : A belief, and the encoded clue.
        state_dim : A node's recurrent state.
        hidden : The hidden layer of the message encoder.
        y_dim : The answer trace.

    Example
    -------
    >>> Widths(15, 39, 76, 20)
    Widths(dim=15, state_dim=39, hidden=76, y_dim=20)
    """
    dim: int
    state_dim: int
    hidden: int
    y_dim: int


def as_widths(widths) -> Widths:
    """
    :class:`Widths` from a :class:`Widths`, a tuple or its ``dict``.

    Parameters:
        widths : The four widths, in order or by name.

    Example
    -------
    >>> as_widths((15, 39, 76, 20)) == as_widths(
    ...     {"dim": 15, "state_dim": 39, "hidden": 76, "y_dim": 20})
    True
    """
    return Widths(**widths) if isinstance(widths, dict) else Widths(*widths)


def count(module: nn.Module) -> int:
    """
    The trainable parameters of a module.

    Parameters:
        module : The module to count.

    Example
    -------
    >>> count(torch.nn.Linear(3, 2))  # doctest: +EXTRA
    8
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def gru_cell(cell: nn.GRUCell, inp: torch.Tensor,
             hidden: torch.Tensor) -> torch.Tensor:
    """
    ``cell(inp, hidden)`` in plain tensor ops, i.e. with a batching rule for
    ``torch.func.vmap``, which the fused kernel behind ``nn.GRUCell`` lacks.

    It is not bit-equal to ``nn.GRUCell``: a network chooses one or the
    other per module, with ``explicit_gru``.

    Parameters:
        cell : The cell whose weights to use.
        inp : The input, ``(batch, input_size)``.
        hidden : The hidden state, ``(batch, hidden_size)``.

    Example
    -------
    >>> cell = torch.nn.GRUCell(3, 2)  # doctest: +EXTRA
    >>> inp, hidden = torch.ones(4, 3), torch.zeros(4, 2)
    >>> torch.allclose(gru_cell(cell, inp, hidden), cell(inp, hidden))
    True
    """
    gi = F.linear(inp, cell.weight_ih, cell.bias_ih)
    gh = F.linear(hidden, cell.weight_hh, cell.bias_hh)
    return gru(gi, gh, hidden)


def gru(gi: torch.Tensor, gh: torch.Tensor,
        hidden: torch.Tensor) -> torch.Tensor:
    """
    The update of a GRU from its input gates ``gi`` and hidden gates ``gh``,
    each the reset, update and new gates side by side: what
    :func:`gru_cell` computes once the gates are computed, in that order.

    Parameters:
        gi : The input gates, ``(..., 3 * hidden_size)``.
        gh : The hidden gates, ``(..., 3 * hidden_size)``.
        hidden : The hidden state, ``(..., hidden_size)``.

    Example
    -------
    >>> zero = torch.zeros(1, 6)  # doctest: +EXTRA
    >>> gru(zero, zero, torch.ones(1, 2)).tolist()
    [[0.5, 0.5]]
    """
    gate_r, gate_z, gate_n = gi.chunk(3, -1)
    hidden_r, hidden_z, hidden_n = gh.chunk(3, -1)
    reset = torch.sigmoid(gate_r + hidden_r)
    keep = torch.sigmoid(gate_z + hidden_z)
    fresh = torch.tanh(gate_n + reset * hidden_n)
    return fresh + keep * (hidden - fresh)


class Site(nn.Module):
    """
    The node module on per-port messages: ``encode = Linear(state_dim + dim
    -> hidden), ReLU, Linear(hidden -> hidden)`` on [own state, message] at
    every port, the mean of the encodings over the ports, then :meth:`fire`:
    ``update = GRUCell(hidden + dim + y_dim -> state_dim)`` on [pool, clue,
    answer], ``norm = LayerNorm(state_dim)`` and ``emit = Linear(state_dim
    -> dim)``, one belief out.

    The modules are built, and so draw their initial weights, in the order
    ``encode``, ``update``, ``norm``, ``emit``. The columns of
    ``encode[0].weight`` are the state then the message; those of
    ``update.weight_ih`` the pool, the clue then the answer.

    There are two operation orders with the same arithmetic. The *reference*
    order runs ``encode`` on the concatenation at every port, then pools.
    The *split* order splits the first map into its state half, once per
    node with the bias, and its message half, once per port, and runs the
    second map after the pool, which it commutes with since it is affine.
    The two differ in rounding.

    Parameters:
        w : The widths.

    Example
    -------
    >>> torch.manual_seed(0)  # doctest: +EXTRA +ELLIPSIS
    <torch._C.Generator object at ...>
    >>> w = Widths(4, 6, 8, 5)
    >>> site = Site(w)
    >>> messages, state = torch.rand(2, 3, 2, 4), torch.rand(2, 3, 6)
    >>> clue, answer = torch.rand(2, 3, 4), torch.rand(2, 3, 5)
    >>> reference = site(messages, state, clue, answer)
    >>> site.split = True
    >>> split = site(messages, state, clue, answer)
    >>> all(map(torch.allclose, reference, split))
    True
    """
    explicit_gru: bool = False
    """ Whether to run ``update`` as :func:`gru_cell` rather than itself. """

    split: bool | None = None
    """
    Whether to compute in the split order: ``None`` for the split order on
    CUDA and the reference order elsewhere, the rule the experiments were
    trained under.
    """

    def __init__(self, w: Widths):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(w.state_dim + w.dim, w.hidden), nn.ReLU(),
            nn.Linear(w.hidden, w.hidden))
        self.update = nn.GRUCell(w.hidden + w.dim + w.y_dim, w.state_dim)
        self.norm = nn.LayerNorm(w.state_dim)
        self.emit = nn.Linear(w.state_dim, w.dim)

    @staticmethod
    def pool(encoded: torch.Tensor, rows: int, sites: int, valid=None,
             degree=None) -> torch.Tensor:
        """
        The port mean of the encodings ``(rows * sites, ports, width)``: a
        plain mean, or with a mask the sum over the real ports divided by
        their number.

        Parameters:
            encoded : The encodings.
            rows : The batch size.
            sites : The number of nodes.
            valid : ``None``, or ``(sites, ports)``, one for a real port and
                    zero for a padded one.
            degree : ``(sites, )`` the number of real ports of each node,
                     when ``valid`` is given.

        Example
        -------
        >>> encoded = torch.tensor([[[1.], [3.]]])  # doctest: +EXTRA
        >>> Site.pool(encoded, 1, 1).tolist()
        [[2.0]]
        >>> valid, degree = torch.tensor([[1., 0.]]), torch.tensor([1.])
        >>> Site.pool(encoded, 1, 1, valid, degree).tolist()
        [[1.0]]
        """
        if valid is None:
            return encoded.mean(1)
        n, ports, width = encoded.shape
        mask = valid.reshape(1, sites, ports, 1).expand(
            rows, -1, -1, width).reshape(n, ports, width)
        divisor = degree.reshape(1, sites, 1).expand(
            rows, -1, -1).reshape(n, 1)
        return (encoded * mask).sum(1) / divisor

    def forward(self, messages, state, clue, answer, valid=None,
                degree=None) -> tuple:
        """
        ``(beliefs (B, sites, dim), states (B, sites, state_dim))`` from the
        messages ``(B, sites, ports, dim)`` at every port.

        Parameters:
            messages : The message at every port.
            state : The states ``(B, sites, state_dim)``.
            clue : The encoded clue ``(B, sites, dim)``.
            answer : The answer trace ``(B, sites, y_dim)``.
            valid : The real ports, see :meth:`pool`.
            degree : The number of real ports, see :meth:`pool`.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        # pylint: disable=too-many-locals
        rows, sites, ports, dim = messages.shape
        message = messages.reshape(-1, ports, dim)
        carried = state.reshape(rows * sites, -1)
        split = messages.is_cuda if self.split is None else self.split
        if split:
            first, relu, second = self.encode
            state_dim = carried.shape[-1]
            shared = F.linear(
                carried,
                first.weight[:, :state_dim],  # ty: ignore[not-subscriptable]
                first.bias)  # ty: ignore[invalid-argument-type]
            per_port = F.linear(
                message,
                first.weight[:, state_dim:],  # ty: ignore[not-subscriptable]
                None)
            pooled = second(self.pool(relu(shared.unsqueeze(1) + per_port),
                                      rows, sites, valid, degree))
        else:
            pooled = self.pool(self.encode(torch.cat([
                carried.unsqueeze(1).expand(-1, ports, -1), message], -1)),
                rows, sites, valid, degree)
        return self.fire(pooled, carried, clue, answer, rows, sites, dim)

    def fire(self, pooled, carried, clue, answer, rows: int, sites: int,
             dim: int) -> tuple:
        """
        The tail every site shares: ``update`` on [pool, clue, answer] and
        the carried state, ``norm``, and ``emit``.

        Parameters:
            pooled : The pooled encodings ``(rows * sites, hidden)``.
            carried : The states ``(rows * sites, state_dim)``.
            clue : The encoded clue ``(rows, sites, dim)``.
            answer : The answer trace ``(rows, sites, y_dim)``.
            rows : The batch size.
            sites : The number of nodes.
            dim : The width of a belief.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        inp = torch.cat([pooled, clue.reshape(rows * sites, -1),
                         answer.reshape(rows * sites, -1)], -1)
        updated = self.norm(
            gru_cell(self.update, inp, carried) if self.explicit_gru
            else self.update(inp, carried))
        return (self.emit(updated).reshape(rows, sites, dim),
                updated.reshape(rows, sites, -1))


class PeerSite(Site):
    """
    The node module wired straight to its peers: the same modules, so the
    same parameters and draws, reading each peer's last belief through a
    gather table ``peers`` ``(sites, degree)``.

    It has one operation order on every device: the first map's state half
    once per node with the bias, its belief half once per *emitting* node,
    gathered onto the ports after the projection, ReLU, the mean over the
    ports, and the second map after the mean; then :meth:`Site.fire`.

    Parameters:
        w : The widths.

    Example
    -------
    >>> w = Widths(4, 6, 8, 5)  # doctest: +EXTRA
    >>> peers = torch.tensor([[1, 2], [0, 2], [0, 1]])
    >>> units, state = torch.rand(2, 3, 4), torch.rand(2, 3, 6)
    >>> clue, answer = torch.rand(2, 3, 4), torch.rand(2, 3, 5)
    >>> site = PeerSite(w)
    >>> belief, _ = site(units, state, clue, answer, peers)
    >>> reference, _ = Site.forward(
    ...     site, units[:, peers], state, clue, answer)
    >>> torch.allclose(belief, reference)
    True
    """
    def forward(self, units, state, clue, answer, peers) -> tuple:
        """
        ``(beliefs (B, sites, dim), states (B, sites, state_dim))`` from the
        beliefs ``units`` ``(B, sites, dim)`` every node emitted last.

        Parameters:
            units : The last beliefs.
            state : The states ``(B, sites, state_dim)``.
            clue : The encoded clue ``(B, sites, dim)``.
            answer : The answer trace ``(B, sites, y_dim)``.
            peers : The peers of every node, ``(sites, degree)`` long.
        """
        # pylint: disable=arguments-differ,too-many-locals
        rows, sites, dim = units.shape
        first, _, second = self.encode
        carried = state.reshape(rows * sites, -1)
        state_dim = carried.shape[-1]
        shared = F.linear(
            carried,
            first.weight[:, :state_dim],  # ty: ignore[not-subscriptable]
            first.bias)  # ty: ignore[invalid-argument-type]
        per_cell = F.linear(
            units.reshape(rows * sites, dim),
            first.weight[:, state_dim:],  # ty: ignore[not-subscriptable]
            None)
        hidden = per_cell.shape[-1]
        ports = F.relu(shared.view(rows, sites, 1, hidden)
                       + per_cell.view(rows, sites, hidden)[:, peers])
        pooled = second(ports.reshape(rows * sites, peers.shape[1],
                                      hidden).mean(1))
        return self.fire(pooled, carried, clue, answer, rows, sites, dim)


class DeepSet(nn.Module):
    """
    The factor module, Deep Sets on its members' beliefs: ``phi =
    Linear(dim -> hidden), ReLU`` embeds each member, the embeddings are
    summed over the members, each one's own included, and ``rho =
    Linear(dim + hidden -> hidden), ReLU, Linear(hidden -> dim)`` answers
    every member from [its belief, the sum]. It is stateless.

    ``phi`` is built before ``rho``, which is the order of their draws. The
    columns of ``rho[0].weight`` are the member then the sum.

    :meth:`forward` computes it in the reference order on beliefs already
    gathered onto the factor ports. :meth:`parts` and :meth:`pool` are its
    two halves, for a network that runs ``phi`` and ``rho``'s member half
    once per node and pools by reduction (a sum, or a mean when the factor
    is a mean-field readout).

    Parameters:
        w : The widths.

    Example
    -------
    >>> w = Widths(4, 6, 8, 5)  # doctest: +EXTRA
    >>> unit = DeepSet(w)
    >>> beliefs = torch.rand(2, 1, 3, 4)
    >>> embedded, member = unit.parts(beliefs.reshape(3 * 2, 4))
    >>> pooled = unit.pool(embedded.view(2, 3, 8).sum(1)).unsqueeze(1)
    >>> split = unit.rho[2](torch.relu(member.view(2, 3, 8) + pooled))
    >>> torch.allclose(split, unit(beliefs).view(2, 3, 4), atol=1e-6)
    True
    """
    def __init__(self, w: Widths):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(w.dim, w.hidden), nn.ReLU())
        self.rho = nn.Sequential(
            nn.Linear(w.dim + w.hidden, w.hidden), nn.ReLU(),
            nn.Linear(w.hidden, w.dim))

    def forward(self, messages) -> torch.Tensor:
        """
        The replies to the beliefs ``messages`` ``(B, factors, arity,
        dim)``, one per member, the same shape.

        Parameters:
            messages : The beliefs at every factor port.
        """
        rows, units, arity, dim = messages.shape
        message = messages.reshape(-1, arity, dim)
        pooled = self.phi(message).sum(1)
        pooled = pooled.unsqueeze(1).expand(-1, arity, -1)
        return self.rho(torch.cat([message, pooled], -1)).reshape(
            rows, units, arity, dim)

    def parts(self, flat) -> tuple:
        """
        ``(embedded, member)``: ``phi`` and ``rho``'s member half, with
        ``rho``'s bias, of flat beliefs ``(n, dim)``, as one product whose
        rows are ``rho``'s then ``phi``'s (the block order fixes the
        summation order of the backward).

        Parameters:
            flat : The beliefs, ``(n, dim)``.
        """
        dim, hidden = self.phi[0].in_features, self.phi[0].out_features
        weights = [self.rho[0].weight[  # ty: ignore[not-subscriptable]
            :, :dim], self.phi[0].weight]
        biases = [self.rho[0].bias, self.phi[0].bias]
        both = F.linear(
            flat,
            torch.cat(weights, 0),  # ty: ignore[invalid-argument-type]
            torch.cat(biases, 0))  # ty: ignore[invalid-argument-type]
        return F.relu(both[:, hidden:]), both[:, :hidden]

    def pool(self, pooled) -> torch.Tensor:
        """
        ``rho``'s pooled half, without bias, on a reduction of the
        embeddings of :meth:`parts`.

        Parameters:
            pooled : The reduced embeddings, ``(n, hidden)``.
        """
        dim = self.phi[0].in_features
        weight = self.rho[0].weight
        return F.linear(
            pooled, weight[:, dim:], None)  # ty: ignore[not-subscriptable]


class Refresh(nn.Module):
    """
    The answer refresh between cycles, ``norm(module(state, answer))`` with
    ``module = GRUCell(state_dim -> y_dim)`` and ``norm = LayerNorm(y_dim)``.

    Parameters:
        w : The widths.

    Example
    -------
    >>> refresh = Refresh(Widths(4, 6, 8, 5))  # doctest: +EXTRA
    >>> refresh(torch.rand(2, 3, 6), torch.rand(2, 3, 5)).shape
    torch.Size([2, 3, 5])
    """
    explicit_gru: bool = False
    """ Whether to run ``module`` as :func:`gru_cell` rather than itself. """

    def __init__(self, w: Widths):
        super().__init__()
        self.module = nn.GRUCell(w.state_dim, w.y_dim)
        self.norm = nn.LayerNorm(w.y_dim)

    def forward(self, state, answer) -> torch.Tensor:
        """
        The refreshed answer trace ``(B, sites, y_dim)``.

        Parameters:
            state : The states ``(B, sites, state_dim)``.
            answer : The answer trace ``(B, sites, y_dim)``.
        """
        rows, sites, width = answer.shape
        inp = state.reshape(rows * sites, -1)
        carried = answer.reshape(rows * sites, width)
        out = gru_cell(self.module, inp, carried) if self.explicit_gru \
            else self.module(inp, carried)
        return self.norm(out).reshape(rows, sites, width)


class HaltHead(nn.Linear):
    """
    One halting logit per node off the answer trace, starting at zero weight
    and bias ``-5``, so that a network trains at full depth until the head
    has learned something. Its constructor draws the default initial weights
    before overwriting them, so a network that builds it last draws the same
    trunk as one without it.

    Parameters:
        y_dim : The width of the answer trace.

    Example
    -------
    >>> halt = HaltHead(5)  # doctest: +EXTRA
    >>> halt.read(torch.rand(2, 3, 5)).tolist()
    [[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]
    """
    def __init__(self, y_dim: int):
        super().__init__(y_dim, 1)
        with torch.no_grad():
            self.weight.zero_()
            self.bias.fill_(-5.0)

    def read(self, answer) -> torch.Tensor:
        """
        The halting logits ``(rows, sites)``.

        Parameters:
            answer : The answer trace ``(rows, sites, y_dim)``.
        """
        return self(answer).squeeze(-1)


def widths_for(budget: int, build,
               ratios: tuple[float, ...] = (1.0, 2.615, 5.231, 1.308),
               low: int = 4, high: int = 255) -> Widths:
    """
    The widths closest to ``budget`` parameters along the ray ``dim :
    state_dim : hidden : y_dim = ratios``.

    The scan builds a network with ``build`` for every ``dim`` from ``low``
    to ``high``, stopping once the count passes twice the budget, and keeps
    the first closest. Building draws initial weights, so the scan consumes
    the global generator: a recipe that seeds, scans and then builds has
    initial weights that depend on it.

    Parameters:
        budget : The number of parameters to aim for.
        build : The network of given widths, whose :func:`count` is taken.
        ratios : The ray, ``ratios[0]`` standing for ``dim`` itself.
        low : The first ``dim`` tried.
        high : The last ``dim`` tried.

    Example
    -------
    >>> widths_for(  # doctest: +EXTRA
    ...     1000, lambda w: torch.nn.Linear(w.dim, w.state_dim))
    Widths(dim=19, state_dim=50, hidden=99, y_dim=25)
    """
    best, best_size = None, None
    for dim in range(low, high + 1):
        widths = Widths(dim, *(max(1, round(dim * r / ratios[0]))
                               for r in ratios[1:]))
        size = count(build(widths))
        if best_size is None or abs(size - budget) < abs(best_size - budget):
            best, best_size = widths, size
        if size > 2 * budget:
            break
    return best  # ty: ignore[invalid-return-type]
