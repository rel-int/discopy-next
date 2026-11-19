# -*- coding: utf-8 -*-

"""
Recursive message passing: the boxes of :mod:`discopy.neural.cell` fired in
synchronous rounds along the tables of :mod:`discopy.neural.wiring`.

A *round* fires every node once on what its ports hold and routes the
beliefs it emits. A *cycle* is :attr:`Recursion.rounds` rounds followed by a
:class:`~discopy.neural.cell.Refresh` of the answer trace. A supervision step
is :attr:`Recursion.cycles` cycles from :meth:`Recursion.initial`: the
leading :attr:`Recursion.detached` cycles run without a gradient, the clue
is then encoded again *with* one, which is how the encoder trains however
many cycles are detached, and the heads read the answer trace after every
differentiated cycle. :meth:`Recursion.act_step` is the same step run as one
segment of adaptive computation time, from a carried state, with a halting
head read off the detached answer.

Two wirings are given:

* :class:`PeerNet` -- every node reads its peers' last beliefs through a
  gather table, a cell-to-cell hop per round;
* :class:`FactorNet` -- every node reads one reply per factor it belongs to,
  every factor (a :class:`~discopy.neural.cell.DeepSet`) its members' beliefs
  of the round before, so a belief reaches the other members of a factor
  two rounds after it is emitted.

A network builds its modules, and so draws its initial weights, in the
order ``cell``, the factors, ``refresh``, ``encoder``, ``bce``, ``sm``,
``conf``, ``y0`` and, when it halts, ``halt``. Its parameters are named and
ordered as in the experiments it comes from, ``y0`` first, so that its
checkpoints, optimiser groups and gradient norms are theirs.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    State
    PortState
    Recursion
    PeerNet
    FactorNet

Example
-------
>>> import torch  # doctest: +EXTRA
>>> from discopy.neural.wiring import primal
>>> triangle = ((0, 1, 2), )
>>> peer = PeerNet((4, 6, 8, 5), primal(triangle), 3, 2, 3, 1, halt=True)
>>> factor = FactorNet((4, 6, 8, 5), triangle, 3, 2, 3, 1, halt=True)
>>> x = torch.ones(2, 3, 3)
>>> for net in (peer, factor):
...     state, every, halt, final = net.act_step(net.initial(x), x)
...     print(type(state).__name__, len(every), tuple(halt.shape))
State 2 (2, 3)
PortState 2 (2, 3)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from discopy.neural import lattice
from discopy.neural.cell import (
    DeepSet, HaltHead, PeerSite, Refresh, Site, as_widths)
from discopy.neural.wiring import routing


class State(NamedTuple):
    """
    What a :class:`PeerNet` carries, ``(B, sites, width)`` each.

    Parameters:
        units : The beliefs every node emitted last round.
        state : The nodes' recurrent states.
        clue : The encoded clue.
        answer : The answer trace.

    Example
    -------
    >>> State(*"usca")
    State(units='u', state='s', clue='c', answer='a')
    """
    units: torch.Tensor
    state: torch.Tensor
    clue: torch.Tensor
    answer: torch.Tensor


class PortState(NamedTuple):
    """
    What a network on per-port messages carries.

    Parameters:
        cells : ``(B, sites, degree, dim)`` the replies at the nodes' ports.
        units : The beliefs the factors read next: as emitted, ``(B, sites,
                dim)``, in the split layout; gathered onto the factor ports,
                ``(B, factors, arity, dim)``, in the reference one.
        state : ``(B, sites, state_dim)`` the nodes' recurrent states.
        clue : ``(B, sites, dim)`` the encoded clue.
        answer : ``(B, sites, y_dim)`` the answer trace.

    Example
    -------
    >>> PortState(*"pusca")._fields
    ('cells', 'units', 'state', 'clue', 'answer')
    """
    cells: torch.Tensor
    units: torch.Tensor
    state: torch.Tensor
    clue: torch.Tensor
    answer: torch.Tensor


class Recursion(nn.Module):
    """
    A recursive network: one shared site on every node, one shared
    :class:`~discopy.neural.cell.DeepSet` per kind of factor, the answer
    refresh, the clue encoder and the heads, fired by :meth:`round`, which a
    subclass defines together with :meth:`initial`.

    The heads read the answer trace: ``bce``, ``sm`` and ``conf`` are the
    lattice heads of :mod:`discopy.neural.lattice`, ``sm`` alone the readout
    of :meth:`act_step`, and ``halt``, when the network halts, a
    :class:`~discopy.neural.cell.HaltHead`. A halting network runs its GRUs
    as :func:`~discopy.neural.cell.gru_cell`, which ``vmap`` can batch.

    Parameters:
        widths : The :class:`~discopy.neural.cell.Widths`, a tuple or its
                 ``dict``.
        n_channels : The alphabet, the last axis of the clue.
        rounds : The rounds of a cycle.
        cycles : The cycles of a step.
        detached : The leading cycles of a step run without a gradient.
        site : The class of the node module, built on the widths.
        factors : The attribute names of the factor modules, in the order
                  they are built.
        halt : Whether to build a halting head.

    Example
    -------
    The smallest subclass: no message at all, every node fires on its own
    state alone.

    >>> import torch  # doctest: +EXTRA
    >>> class Alone(Recursion):
    ...     def initial(self, x):
    ...         rows, cells = x.shape[:2]
    ...         zeros = torch.zeros(())
    ...         return PortState(
    ...             zeros.expand(rows, cells, 1, self.widths.dim), None,
    ...             zeros.expand(rows, cells, self.widths.state_dim),
    ...             self.encoder(x),
    ...             self.y0.expand(rows, cells, self.widths.y_dim))
    ...     def round(self, s):
    ...         belief, state = self.cell(s.cells, s.state, s.clue, s.answer)
    ...         return s._replace(cells=belief.unsqueeze(2), state=state)
    >>> net = Alone((4, 6, 8, 5), 3, 2, 3, 1, halt=True)
    >>> [name for name, _ in net.named_children()]
    ['cell', 'refresh', 'encoder', 'bce', 'sm', 'conf', 'halt']
    >>> next(net.named_parameters())[0]
    'y0'
    >>> assert net.cell.explicit_gru and net.refresh.explicit_gru
    >>> x = torch.rand(2, 7, 3)
    >>> state, every, halt, final = net.act_step(net.initial(x), x)
    >>> len(every), tuple(final.shape), tuple(halt.shape)
    (2, (2, 7, 3), (2, 7))
    """
    # pylint: disable=too-many-instance-attributes
    deep: bool = True
    """ Whether :meth:`unroll` and :meth:`window` read the heads after every
    differentiated cycle, rather than only after the last. """

    halt_detach: bool = True
    """ Whether :meth:`window` reads the halting head off the detached
    answer, so that the halting loss trains the head alone. """

    fixed_shape: bool = False
    """ Whether :meth:`step` projects at the fixed shape, drawing for every
    row, see :func:`~discopy.neural.lattice.project`. """

    compile_options: dict = {}
    """ The keywords :meth:`compile` passes to ``torch.compile`` unless it
    is given others, e.g. ``dynamic=False`` for a network compiled twice
    at two batch sizes in one process, which would otherwise make the
    second graph dynamic. """

    def __init__(self, widths, n_channels: int, rounds: int, cycles: int,
                 detached: int, site: type = Site, factors: tuple = (),
                 halt: bool = False):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()
        self.widths = w = as_widths(widths)
        self.n_channels = n_channels
        self.rounds, self.cycles, self.detached = rounds, cycles, detached
        self.factors = tuple(factors)
        self.cell = site(w)
        for name in self.factors:
            setattr(self, name, DeepSet(w))
        self.refresh = Refresh(w)
        self.encoder = nn.Linear(n_channels, w.dim)
        self.bce = nn.Linear(w.y_dim, n_channels)
        self.sm = nn.Linear(w.y_dim, n_channels)
        self.conf = nn.Linear(w.y_dim, 1)
        self.y0 = nn.Parameter(torch.zeros(w.y_dim))
        if halt:
            self.halt = HaltHead(w.y_dim)
            self.cell.explicit_gru = self.refresh.explicit_gru = True
        self.compile_options = dict(self.compile_options)
        self.cycler, self.unroller = self.cycle, self.unroll

    def shape(self) -> dict:
        """ What a build needs to rebuild this network. """
        return {"widths": self.widths._asdict(), "rounds": self.rounds,
                "cycles": self.cycles, "detached": self.detached}

    def compile(self, unroll: bool = True, **kwargs) -> Recursion:
        """
        Compile the whole supervision step :meth:`unroll` as one graph, or
        one :meth:`cycle`, which a search or a rollout that returns to the
        host between steps is better served by. The module itself is not
        wrapped, so its parameter names do not change.

        The compiled functions are bound to this network: compile a copy
        of it, rather than copying it compiled.

        Parameters:
            unroll : Whether to compile :meth:`unroll`, else :meth:`cycle`.
            kwargs : Passed to ``torch.compile``, over
                     :attr:`compile_options` and, for the unroll,
                     ``fullgraph=True``.
        """
        # pylint: disable=arguments-differ  # nn.Module.compile wraps forward
        kwargs = {**self.compile_options, **kwargs}
        if unroll:
            kwargs.setdefault("fullgraph", True)
            self.unroller = torch.compile(  # ty: ignore[no-matching-overload]
                self.unroll, **kwargs)
        else:
            self.cycler = torch.compile(  # ty: ignore[no-matching-overload]
                self.cycle, **kwargs)
        return self

    def initial(self, x):
        """
        The state a step starts from: the encoded clue, ``y0`` on the
        answer trace and zeros elsewhere, under ``no_grad``. So ``y0``
        never receives a gradient and stays at zero: it is a parameter so
        that checkpoints keep their names.

        Parameters:
            x : The clue, ``(B, sites, n_channels)``.
        """
        raise NotImplementedError

    def round(self, s):
        """
        One synchronous round: every node fires, then every port routes.

        Parameters:
            s : The carried state.
        """
        raise NotImplementedError

    def cycle(self, s):
        """
        :attr:`rounds` rounds, then the answer refresh.

        Parameters:
            s : The carried state.
        """
        for _ in range(self.rounds):
            s = self.round(s)
        return s._replace(answer=self.refresh(s.state, s.answer))

    def heads(self, answer):
        """
        The three lattice heads read off the answer trace, as
        :class:`discopy.neural.lattice.Heads`.

        Parameters:
            answer : The answer trace.
        """
        return lattice.Heads(
            self.bce(answer), self.sm(answer), self.conf(answer))

    def forward(self, x, grad: bool = True) -> tuple:
        """
        :meth:`run`, so that calling the network runs a supervision step.

        Parameters:
            x : The clue.
            grad : Whether to differentiate the window.
        """
        return self.run(x, grad)

    def run(self, x, grad: bool = True) -> tuple:
        """
        One supervision step: ``(state, heads, every)``, the state it ended
        on, the last heads and the heads of every differentiated cycle, by
        :meth:`unroll` or its compiled form.

        Parameters:
            x : The clue.
            grad : Whether to differentiate the window.
        """
        return self.unroller(x, grad)

    def unroll(self, x, grad: bool = True) -> tuple:
        """
        :meth:`run` in eager PyTorch, what :meth:`compile` compiles.

        Parameters:
            x : The clue.
            grad : Whether to differentiate the window.
        """
        state = self.initial(x)
        with torch.no_grad():
            for _ in range(self.detached):
                state = self.cycler(state)
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            state = state._replace(clue=self.encoder(x))
            every = []
            for _ in range(self.cycles - self.detached):
                state = self.cycler(state)
                if self.deep:
                    every.append(self.heads(state.answer))
            heads = every[-1] if every else self.heads(state.answer)
        return state, heads, tuple(every)

    def prefix(self, s):
        """
        The leading :attr:`detached` cycles of a segment, forward only.

        Parameters:
            s : The carried state.
        """
        with torch.no_grad():
            for _ in range(self.detached):
                s = self.cycler(s)
        return s

    def window(self, s, x, cycle=None) -> tuple:
        """
        The rest of a segment: ``(state, every, halt)``, the clue encoded
        again from ``x``, the differentiated cycles with an ``sm`` readout
        after each, and the halting logits off the final answer.

        Parameters:
            s : The state :meth:`prefix` returned.
            x : The clue.
            cycle : The cycle to run, by default :meth:`cycle` or its
                    compiled form; the eager one under ``vmap``.
        """
        if not hasattr(self, "halt"):
            raise AttributeError(f"{type(self).__name__} has no halting "
                                 "head: build it with halt=True")
        s = s._replace(clue=self.encoder(x))
        cycle = cycle or self.cycler
        every = []
        for _ in range(self.cycles - self.detached):
            s = cycle(s)
            if self.deep:
                every.append(self.sm(s.answer))
        if not every:
            every.append(self.sm(s.answer))
        answer = s.answer.detach() if self.halt_detach else s.answer
        return s, tuple(every), self.halt.read(answer)

    def act_step(self, s, x, grad: bool = True) -> tuple:
        """
        One segment of adaptive computation time from the carried state
        ``s``: ``(state, every, halt, final)``, the state it ended on, still
        attached, the readouts of the differentiated cycles, the halting
        logits and the last readout.

        Parameters:
            s : The carried state.
            x : The clue.
            grad : Whether to differentiate the window.
        """
        grad = grad and torch.is_grad_enabled()
        s = self.prefix(s)
        with torch.set_grad_enabled(grad):
            s, every, halt = self.window(s, x)
        return s, every, halt, every[-1]

    @torch.no_grad()
    def step(self, x, given, generator=None, theta: float = lattice.THETA,
             theta_cls: float = lattice.THETA_CLS,
             temp: float = lattice.TEMP) -> lattice.Projection:
        """
        One supervision step from :meth:`initial` plus the projection of
        :func:`~discopy.neural.lattice.project`, at the fixed shape when
        :attr:`fixed_shape`: one model call of the chain search.

        Parameters:
            x : The lattice.
            given : The given cells.
            generator : The generator of the decisions.
            theta : The kill threshold.
            theta_cls : The conflict threshold.
            temp : The temperature of the decisions.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _, heads, _ = self.run(x, grad=False)
        return lattice.project(
            x, given, heads, theta, theta_cls, temp, generator,
            fixed_shape=self.fixed_shape)


class PeerNet(Recursion):
    """
    A recursive network wired straight from node to node: one
    :class:`~discopy.neural.cell.PeerSite` reads the last beliefs of every
    node's peers through the table ``peers``, which every node fills to the
    same degree, so that the pool is a plain mean.

    Parameters:
        widths : The widths.
        peers : ``(sites, degree)`` the peers of every node, e.g. the
                :func:`~discopy.neural.wiring.primal` graph of a factor
                table.
        n_channels : The alphabet.
        rounds : The rounds of a cycle.
        cycles : The cycles of a step.
        detached : The leading cycles without a gradient.
        halt : Whether to build a halting head.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> net = PeerNet((4, 6, 8, 5), ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1)
    >>> s = net.initial(torch.ones(2, 3, 3))
    >>> tuple(net.round(s).units.shape)
    (2, 3, 4)
    """
    def __init__(self, widths, peers, n_channels: int, rounds: int,
                 cycles: int, detached: int, halt: bool = False):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__(widths, n_channels, rounds, cycles, detached,
                         PeerSite, (), halt)
        self.register_buffer("peers", torch.as_tensor(peers, dtype=torch.long),
                             persistent=False)

    @torch.no_grad()
    def initial(self, x) -> State:
        rows, cells = x.shape[:2]
        w = self.widths
        zeros = torch.zeros((), dtype=x.dtype, device=x.device)
        return State(zeros.expand(rows, cells, w.dim),
                     zeros.expand(rows, cells, w.state_dim),
                     self.encoder(x), self.y0.expand(rows, cells, w.y_dim))

    def round(self, s: State) -> State:
        units, state = self.cell(s.units, s.state, s.clue, s.answer,
                                 self.peers)
        return s._replace(units=units, state=state)


class FactorNet(Recursion):
    """
    A recursive network on the factor graph of a factor table: one
    :class:`~discopy.neural.cell.Site` serves every node, reading one reply
    per factor it belongs to, and one :class:`~discopy.neural.cell.DeepSet`,
    named ``factor``, serves every factor, reading its members' beliefs of
    the round before. A node in fewer factors than the largest degree has
    its ports padded and masked, see :func:`~discopy.neural.wiring.routing`.

    There are two layouts of the factors' half of the round, chosen by
    :attr:`split`. The *reference* layout gathers the beliefs onto the factor
    ports and runs :meth:`DeepSet.forward <discopy.neural.cell.DeepSet>`;
    the *split* layout keeps the beliefs as emitted and runs
    :meth:`replies`.

    Parameters:
        widths : The widths.
        units : The factor table, every factor of the same arity.
        n_channels : The alphabet.
        rounds : The rounds of a cycle.
        cycles : The cycles of a step.
        detached : The leading cycles without a gradient.
        pool : ``None``, or a function summing a per-node ``(B, sites, h)``
               tensor over every factor's members, ``(B, factors, h)``, by
               reductions rather than a gather; the split layout uses it.
        n_nodes : The number of nodes, by default one more than the
                  largest.
        factor : The attribute name of the factor module.
        halt : Whether to build a halting head.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> path = FactorNet((4, 6, 8, 5), ((0, 1), (1, 2)), 3, 2, 3, 1)
    >>> path.real_degree.tolist(), path.to_cells.tolist()
    ([1.0, 2.0, 1.0], [0, 0, 1, 2, 3, 3])
    >>> s = path.initial(torch.ones(2, 3, 3))
    >>> tuple(s.units.shape), tuple(path.round(s).cells.shape)
    ((2, 2, 2, 4), (2, 3, 2, 4))
    >>> path.split = True
    >>> tuple(path.round(path.initial(torch.ones(2, 3, 3))).units.shape)
    (2, 3, 4)
    """
    split: bool | None = None
    """
    Whether to run the split layout: ``None`` for the split layout on CUDA
    and the reference one elsewhere, the rule the experiments were trained
    under.
    """

    def __init__(self, widths, units, n_channels: int, rounds: int,
                 cycles: int, detached: int, pool=None,
                 n_nodes: int | None = None, factor: str = "unit",
                 halt: bool = False):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__(widths, n_channels, rounds, cycles, detached,
                         Site, (factor, ), halt)
        graph = routing(units, n_nodes)
        self.factor, self.pool = factor, pool
        self.degree, self.arity = graph.degree, graph.arity
        self.n_cells = len(graph.degrees)
        self.n_units = len(graph.to_factors) // graph.arity
        padded = graph.valid is not None
        self.register_buffer(
            "to_units", torch.tensor(graph.to_factors), persistent=False)
        self.register_buffer(
            "to_cells", torch.tensor(graph.to_nodes), persistent=False)
        self.register_buffer(
            "port_valid", torch.tensor(graph.valid) if padded else None,
            persistent=False)
        self.register_buffer(
            "real_degree", torch.tensor(graph.degrees, dtype=torch.float32)
            if padded else None, persistent=False)

    def shape(self) -> dict:
        """ :meth:`Recursion.shape` plus ``graph: "fg"``. """
        return {**super().shape(), "graph": "fg"}

    def splits(self, tensor) -> bool:
        """
        Whether the split layout runs on tensors like ``tensor``.

        Parameters:
            tensor : A tensor of the state.
        """
        return tensor.is_cuda if self.split is None else self.split

    @torch.no_grad()
    def initial(self, x) -> PortState:
        rows, w = len(x), self.widths
        zeros = torch.zeros((), dtype=x.dtype, device=x.device)
        units = (rows, self.n_cells, w.dim) if self.splits(x) \
            else (rows, self.n_units, self.arity, w.dim)
        return PortState(
            zeros.expand(rows, self.n_cells, self.degree, w.dim),
            zeros.expand(*units),
            zeros.expand(rows, self.n_cells, w.state_dim),
            self.encoder(x),
            self.y0.expand(rows, self.n_cells, w.y_dim))

    def round(self, s: PortState) -> PortState:
        belief, state = self.cell(s.cells, s.state, s.clue, s.answer,
                                  self.port_valid, self.real_degree)
        rows, _, dim = belief.shape
        if self.splits(belief):
            reply = self.replies(s.units)
            units = belief
        else:
            reply = getattr(self, self.factor)(s.units)
            units = belief[:, self.to_units].view(rows, self.n_units,
                                                  self.arity, dim)
        replies = reply.reshape(rows, self.n_units * self.arity, dim)
        cells = replies[  # ty: ignore[invalid-argument-type]
            :, self.to_cells].view(rows, self.n_cells, self.degree, dim)
        return PortState(cells, units, state, s.clue, s.answer)

    def replies(self, belief) -> torch.Tensor:
        """
        The factor module on the beliefs as emitted, ``(B, sites, dim)``,
        in the split layout: ``(B, factors, arity, dim)`` replies. ``phi``
        and ``rho``'s member half run once per node rather than once per
        port, the embeddings are summed per factor, by :attr:`pool` when
        given, and only ``rho``'s member half is gathered onto the ports.

        Parameters:
            belief : The beliefs every node emitted.
        """
        rows, cells, dim = belief.shape
        unit = getattr(self, self.factor)
        hidden = unit.phi[0].out_features
        embedded, member = unit.parts(belief.reshape(rows * cells, dim))
        embedded = embedded.view(rows, cells, hidden)
        summed = self.pool(embedded) if self.pool is not None \
            else embedded[:, self.to_units].view(
                rows, self.n_units, self.arity, hidden).sum(2)
        shared = unit.pool(summed.reshape(rows * self.n_units, hidden))
        per_port = member.view(rows, cells, hidden)[:, self.to_units]
        return unit.rho[2](F.relu(
            per_port.view(rows, self.n_units, self.arity, hidden)
            + shared.view(rows, self.n_units, 1, hidden)))
