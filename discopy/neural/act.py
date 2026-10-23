# -*- coding: utf-8 -*-

"""
Adaptive computation time (ACT) for a recursive network: the recursion run
in *segments* from a carried state, a halt head deciding after every
segment whether the answer is final.

A segment starts from a carried state: its leading ``detached`` cycles run
forward only, the clue is re-encoded with a gradient, the remaining cycles
are differentiated with a readout after each, and the per-cell halt logits
are read off the final, detached answer. A board's halt logit is their soft
minimum (:func:`halt_logit`).

* :func:`terms` and :func:`nearest_terms` -- the loss of one segment: the
  deep cross-entropy over the readouts plus the halt BCE against the
  correctness of the final readout (:func:`halt_terms`), with one target
  or the nearest of several admissible ones.
* :class:`ActPool` -- ``batch_size`` persistent slots, one segment and one
  optimizer step per iteration; a slot whose board halt logit turns
  positive, or which reaches ``n_sup`` segments, gets a fresh puzzle. Every
  board's gradient passes the per-board clip of
  :func:`~discopy.neural.train.per_board_backward`, then one global norm
  clip, a spike watchdog and the optimizer.
* :func:`rollout` -- the evaluation: one trajectory of segments, read at
  every cap as the fixed-compute answer, the ACT depth and the most
  confident answer so far.

The pool reads its net through ``initial(x)``, ``prefix(s)``,
``window(s, x, cycle=...)``, ``cycle`` and the net's ``slot_clip`` and
``slot_ref_running`` attributes, which it sets, and :func:`rollout`
through ``initial(x)`` and ``act_step(s, x)``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    ActConfig
    ActPool

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        halt_logit
        decode
        detach_state
        where_state
        halt_terms
        deep_ce
        terms
        nearest_terms
        rollout

Example
-------
>>> halt = torch.tensor([[-1., 3.], [2., 3.]])  # doctest: +EXTRA
>>> halt_logit(halt) > 0
tensor([False,  True])
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from discopy.neural.train import per_board_backward, positive, update_reference


def halt_logit(halt) -> torch.Tensor:
    """
    The board's halt logit, the soft minimum ``-logsumexp(-q)`` of the
    per-cell logits ``q``.

    Parameters:
        halt : The per-cell halt logits ``(rows, cells)``.

    Example
    -------
    >>> halt_logit(torch.tensor([[0., 0.]]))  # doctest: +EXTRA
    tensor([-0.6931])
    """
    return -torch.logsumexp(-halt, dim=-1)


def decode(logits, given, x0) -> torch.Tensor:
    """
    The readout's argmax ``(rows, cells)`` with the givens written back.

    Parameters:
        logits : The readout ``(rows, cells, channels)``.
        given : Which cells are given ``(rows, cells)``.
        x0 : The puzzle lattice ``(rows, cells, channels)``.

    Example
    -------
    >>> x0 = torch.tensor([[[0., 1.], [1., 1.]]])  # doctest: +EXTRA
    >>> logits = torch.tensor([[[2., 0.], [0., 2.]]])
    >>> decode(logits, x0.sum(-1) == 1, x0)
    tensor([[1, 1]])
    """
    return torch.where(given, x0.argmax(-1), logits.argmax(-1))


def detach_state(s):
    """
    Every field of a state, detached, as a state of the same type.

    Parameters:
        s : A named tuple of tensors.

    Example
    -------
    >>> from collections import namedtuple  # doctest: +EXTRA
    >>> State = namedtuple("State", "units answer")
    >>> units = torch.ones(1, requires_grad=True)
    >>> s = detach_state(State(units, torch.ones(1)))
    >>> type(s).__name__, s.units.requires_grad
    ('State', False)
    """
    return type(s)(*(t.detach() for t in s))


def where_state(mask, fresh, kept):
    """
    ``fresh`` on the rows where ``mask`` ``(rows, )`` holds, ``kept``
    elsewhere, field by field, as a state of the type of ``fresh``.

    Parameters:
        mask : Which rows to take from ``fresh``.
        fresh : A named tuple of tensors, the rows first.
        kept : A named tuple of tensors of the same shapes.

    Example
    -------
    >>> from collections import namedtuple  # doctest: +EXTRA
    >>> State = namedtuple("State", "units answer")
    >>> fresh, kept = State(torch.zeros(2, 1), torch.zeros(2)), State(
    ...     torch.ones(2, 1), torch.ones(2))
    >>> where_state(torch.tensor([True, False]), fresh, kept)
    State(units=tensor([[0.],
            [1.]]), answer=tensor([0., 1.]))
    """
    return type(fresh)(*(
        torch.where(mask.reshape(-1, *([1] * (a.dim() - 1))), a, b)
        for a, b in zip(fresh, kept)))


def halt_terms(ce_cells, ce, halt, cell_ok, halt_weight: float = 1.0,
               board_weight: float = 1.0) -> tuple:
    """
    The halt loss of a segment and its total: ``(loss, ce, q, cell_ok,
    board)`` with ``q`` the per-cell halt BCE against ``cell_ok`` plus
    ``board_weight`` x the BCE of the board halt logit against "every cell
    correct", ``loss = ce + halt_weight * q`` and ``board`` ``(rows, )``
    each board's own detached loss, what the per-board clip is relative to.

    Parameters:
        ce_cells : The per-cell cross-entropy ``(rows * cells, )``.
        ce : Its mean.
        halt : The per-cell halt logits ``(rows, cells)``.
        cell_ok : Which cells the final readout gets right.
        halt_weight : The weight of the halt loss.
        board_weight : The weight of its board term, within it.

    Example
    -------
    >>> ce_cells = torch.tensor([1., 1., 3., 3.])  # doctest: +EXTRA
    >>> halt, cell_ok = torch.zeros(2, 2), torch.tensor([[1, 1], [0, 1]]) > 0
    >>> loss, ce, q, _, board = halt_terms(ce_cells, ce_cells.mean(), halt,
    ...                                    cell_ok, board_weight=0.)
    >>> loss, board
    (tensor(2.6931), tensor([1.6931, 3.6931]))
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rows = cell_ok.shape[0]
    q_cells = F.binary_cross_entropy_with_logits(
        halt, cell_ok.detach().to(halt.dtype), reduction="none")
    q = q_cells.mean()
    q_board = None
    if board_weight:
        q_board = F.binary_cross_entropy_with_logits(
            halt_logit(halt).float(), cell_ok.all(-1).to(torch.float32),
            reduction="none")
        q = q + board_weight * q_board.mean()
    with torch.no_grad():
        board = ce_cells.reshape(rows, -1).mean(1) \
            + halt_weight * q_cells.reshape(rows, -1).float().mean(1)
        if q_board is not None:
            board = board + halt_weight * board_weight * q_board.float()
    return ce + halt_weight * q, ce, q, cell_ok, board.detach()


def deep_ce(every, target) -> tuple:
    """
    The cross-entropy of every readout against ``target``, averaged over
    the readouts: ``(ce_cells, ce)``, per cell and over all of them.

    Parameters:
        every : The readouts, each ``(rows, cells, n)``.
        target : The symbol of every cell, ``(rows, cells)``.

    Example
    -------
    >>> logits = torch.zeros(1, 2, 4)  # doctest: +EXTRA
    >>> ce_cells, ce = deep_ce((logits, logits), torch.tensor([[0, 3]]))
    >>> round(float(ce), 4)
    1.3863
    """
    n = every[0].shape[-1]
    flat = target.reshape(-1)
    ce_cells = sum(F.cross_entropy(logits.reshape(-1, n).float(), flat,
                                   reduction="none")
                   for logits in every) / len(every)
    return ce_cells, ce_cells.mean()


def terms(every, halt, final, x, target, given, halt_weight: float = 1.0,
          board_weight: float = 1.0) -> tuple:
    """
    The loss of a segment with one target: ``(loss, ce, q, cell_ok,
    board)``, the deep cross-entropy over the readouts against ``target``
    plus the halt loss of :func:`halt_terms` against the correctness of
    ``final``, decoded with the givens written back.

    Parameters:
        every : The readouts ``(rows, cells, channels)`` of the window.
        halt : The per-cell halt logits ``(rows, cells)``.
        final : The readout the halt head is graded on.
        x : The puzzle lattice ``(rows, cells, channels)``.
        target : The solution ``(rows, cells)``.
        given : Which cells are given ``(rows, cells)``.
        halt_weight : The weight of the halt loss.
        board_weight : The weight of its board term, within it.

    Example
    -------
    >>> x = torch.ones(1, 2, 2)  # doctest: +EXTRA
    >>> every = (torch.tensor([[[9., 0.], [0., 9.]]]), )
    >>> loss, ce, q, cell_ok, board = terms(
    ...     every, torch.full((1, 2), 5.), every[-1], x,
    ...     torch.tensor([[0, 1]]), x.sum(-1) == 1)
    >>> cell_ok, round(float(loss), 4), board.shape
    (tensor([[True, True]]), 0.0202, torch.Size([1]))
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ce_cells, ce = deep_ce(every, target)
    with torch.no_grad():
        cell_ok = decode(final, given, x) == target
    return halt_terms(ce_cells, ce, halt, cell_ok, halt_weight, board_weight)


def nearest_terms(every, halt, final, x, sols, mask, given,
                  halt_weight: float = 1.0,
                  board_weight: float = 1.0) -> tuple:
    """
    The loss of a segment with several admissible solutions: the
    cross-entropy against the solution the readouts give the least negative
    log-likelihood, averaged over the cells, and the halt loss against the
    agreement of ``final`` with the solution it agrees with on the most
    cells. Returns ``(loss, ce, q, cell_ok, board)`` as :func:`terms`.

    Parameters:
        every : The readouts ``(rows, cells, channels)`` of the window.
        halt : The per-cell halt logits ``(rows, cells)``.
        final : The readout the halt head is graded on.
        x : The puzzle lattice ``(rows, cells, channels)``.
        sols : The admissible solutions ``(rows, K, cells)``.
        mask : Which of the ``K`` solutions are real ``(rows, K)``.
        given : Which cells are given ``(rows, cells)``.
        halt_weight : The weight of the halt loss.
        board_weight : The weight of its board term, within it.

    Example
    -------
    >>> x = torch.ones(1, 2, 2)  # doctest: +EXTRA
    >>> every = (torch.tensor([[[0., 9.], [9., 0.]]]), )
    >>> sols = torch.tensor([[[0, 1], [1, 0]]])
    >>> _, ce, _, cell_ok, _ = nearest_terms(
    ...     every, torch.zeros(1, 2), every[-1], x, sols,
    ...     torch.tensor([[True, True]]), x.sum(-1) == 1)
    >>> cell_ok, ce < 1e-3
    (tensor([[True, True]]), tensor(True))
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    n = every[0].shape[-1]
    rows, _, cells = sols.shape
    index = torch.arange(rows, device=sols.device)
    with torch.no_grad():
        logp = sum(F.log_softmax(logits.reshape(-1, n).float(), -1)
                   for logits in every) / len(every)
        per_k = -logp.view(rows, cells, n).gather(
            2, sols.permute(0, 2, 1)).mean(1)
        choice = per_k.masked_fill(~mask, float("inf")).argmin(1)
    target = sols[index, choice]
    ce_cells, ce = deep_ce(every, target)
    with torch.no_grad():
        agree = (decode(final, given, x).unsqueeze(1) == sols) \
            & mask.unsqueeze(-1)
        nearest = agree.sum(-1).argmax(1)
        cell_ok = agree[index, nearest]
    return halt_terms(ce_cells, ce, halt, cell_ok, halt_weight, board_weight)


@dataclass
class ActConfig:
    """
    The ACT training protocol; the defaults are the sudoku recipe of
    arXiv:2610.18649, its Table 7.

    Parameters:
        steps : The optimizer steps, one segment each.
        batch_size : The slots.
        n_sup : The segment cap of one puzzle.
        lr : The peak learning rate.
        weight_decay : The decay of the matrices.
        warmup_frac : The fraction of the steps the warmup takes.
        betas : The moment decays of AdamW.
        lr_floor : The final fraction of the peak learning rate.
        ema_decay : The decay of the weight average.
        grad_clip : The global norm clip.
        slot_clip : The per-board clip factor ``k``, ``None`` for no
            per-board clip.
        slot_ref_decay : The decay of its slow reference.
        halt_weight : The weight of the halt loss.
        halt_board_weight : The weight of its board term.
        halt_threshold : The board halt logit a slot halts above.
        skip_spike : The pre-clip norm above which a step is rejected,
            ``None`` for no watchdog.
        stagger_pool : Whether slot ``i`` starts at depth ``i mod n_sup``.

    Example
    -------
    >>> ActConfig(steps=10).batch_size
    512
    """
    # pylint: disable=too-many-instance-attributes  # one per knob
    steps: int = 275000
    batch_size: int = 512
    n_sup: int = 32
    lr: float = 7.5e-4
    weight_decay: float = 0.04821
    warmup_frac: float = 0.10
    betas: tuple = (0.9, 0.95)
    lr_floor: float = 0.05
    ema_decay: float = 0.9995
    grad_clip: float = 0.5
    slot_clip: float = 50.0
    slot_ref_decay: float = 0.99
    halt_weight: float = 1.0
    halt_board_weight: float = 1.0
    halt_threshold: float = 0.0
    skip_spike: float | None = 50.0
    stagger_pool: bool = True


class ActPool:
    """
    The slot pool and its one-segment-per-optimizer-step iteration.

    The slots hold what the stream's ``take(n)`` returns, a puzzle lattice
    ``x`` and a solution lattice ``y`` per slot (:meth:`slots`), and what is
    read off them (:meth:`load`): the ``target`` symbols and the ``given``
    cells. The per-board clip state lives on the net: ``slot_clip`` (the
    factor) and ``slot_ref_running`` (the slow reference), where a custom
    backward can read it.

    Parameters:
        net : The recursive network.
        stream : Where fresh puzzles come from, by ``take(n)``.
        cfg : The protocol.

    Note
    ----
    A variant of the protocol overrides a method, never the op order:
    :meth:`segment` (the gradient of one segment), :meth:`board_loss`,
    :meth:`window_loss`, :meth:`targets`, :meth:`slots` and :meth:`load`.

    Example
    -------
    >>> from collections import namedtuple  # doctest: +EXTRA
    >>> State = namedtuple("State", "state answer")
    >>> class Net(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.cell = torch.nn.Linear(2, 2)
    ...         self.sm = torch.nn.Linear(2, 2)
    ...         self.halt = torch.nn.Linear(2, 1)
    ...     def initial(self, x):
    ...         return State(x, torch.zeros_like(x))
    ...     def cycle(self, s):
    ...         answer = torch.tanh(self.cell(s.state) + s.answer)
    ...         return State(s.state, answer)
    ...     def prefix(self, s):
    ...         with torch.no_grad():
    ...             return self.cycle(s)
    ...     def window(self, s, x, cycle=None):
    ...         s = (cycle or self.cycle)(s)
    ...         every = (self.sm(s.answer), )
    ...         return s, every, self.halt(s.answer.detach()).squeeze(-1)
    >>> class Stream:
    ...     def take(self, n):
    ...         y = torch.nn.functional.one_hot(torch.arange(n * 3) % 2, 2)
    ...         return torch.ones(n, 3, 2), y.float().view(n, 3, 2)
    >>> _ = torch.manual_seed(0)
    >>> net = Net()
    >>> pool = ActPool(net, Stream(), ActConfig(batch_size=4, n_sup=2))
    >>> optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    >>> scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    >>> from discopy.neural.train import EMA
    >>> ema = EMA(net, .9)
    >>> stats = pool.iterate(optimizer, scheduler, ema)
    >>> int(stats["halted"]), pool.depth.tolist()
    (2, [1, 0, 1, 0])
    """
    # pylint: disable=too-many-instance-attributes  # one per slot field
    def __init__(self, net, stream, cfg: ActConfig):
        self.net, self.stream, self.cfg = net, stream, cfg
        net.slot_clip = positive(cfg.slot_clip)
        net.slot_ref_running = None
        self.load(*(t.clone() for t in stream.take(cfg.batch_size)))
        with torch.no_grad():
            self.state = net.initial(self.x)
        device = self.x.device
        self.depth = torch.zeros(cfg.batch_size, dtype=torch.long,
                                 device=device)
        if cfg.stagger_pool:
            self.depth = (torch.arange(cfg.batch_size, device=device)
                          % cfg.n_sup).long()

    def slots(self) -> tuple:
        """ What the stream gave the slots: ``(x, y)``. """
        return self.x, self.y

    def load(self, x, y) -> None:
        """
        Fill the slots with ``x`` and ``y``, and read the ``target`` symbols
        and the ``given`` cells off them.

        Parameters:
            x : The puzzle lattices ``(rows, cells, channels)``.
            y : The solution lattices ``(rows, cells, channels)``.
        """
        self.x, self.y = x, y
        self.target = self.y.argmax(-1)
        self.given = self.x.sum(-1) == 1

    def targets(self) -> tuple:
        """ What a board's loss reads besides its state: ``(x, target,
        given)``. """
        return self.x, self.target, self.given

    def window_loss(self, every, halt, x, *labels) -> tuple:
        """
        The loss of a window, :func:`terms` on the last readout: ``(loss,
        ce, q, cell_ok, board)``. A variant grading the halt head on
        another decoding overrides it.

        Parameters:
            every : The readouts of the window.
            halt : The per-cell halt logits.
            x : The puzzle lattices.
            labels : The rest of :meth:`targets`.
        """
        weights = self.cfg.halt_weight, self.cfg.halt_board_weight
        return terms(
            every, halt, every[-1], x,
            *labels, *weights)  # ty: ignore[too-many-positional-arguments]

    def board_loss(self, net, *args) -> tuple:
        """
        One board's window and loss, for
        :func:`~discopy.neural.train.per_board_backward`: ``args`` are the
        fields of its state then its :meth:`targets`. Returns the loss and
        ``(ce, q, cell_ok, halt, *state)``, detached.

        Parameters:
            net : The net, its parameters swapped by ``functional_call``.
            args : The board's state fields and targets.
        """
        n = len(self.state)
        s, every, halt = net.window(type(self.state)(*args[:n]), args[n],
                                    cycle=net.cycle)
        loss, ce, q, cell_ok, _ = self.window_loss(every, halt, *args[n:])
        return loss, (ce.detach(), q.detach(), cell_ok, halt.detach(),
                      *(t.detach() for t in s))

    def segment(self) -> tuple:
        """
        One segment on the pool with its clipped gradient in ``.grad``:
        the prefix batched and forward-only, then the window per board
        through :func:`~discopy.neural.train.per_board_backward`, or
        batched with no per-board clip when ``net.slot_clip`` is ``None``.
        Returns ``(state, halt, loss, ce, q, correct, clip statistics)``.
        """
        cfg, net = self.cfg, self.net
        s = net.prefix(self.state)
        if net.slot_clip is None:
            return self.batch_segment(s)
        aux, clip = per_board_backward(
            net, self.board_loss, (*s, *self.targets()), net.slot_clip,
            reference=net.slot_ref_running)
        ce, q, cell_ok, halt, *carried = (a.squeeze(1) if a.dim() > 1 else a
                                          for a in aux)
        loss = ce.mean() + cfg.halt_weight * q.mean()
        return (type(s)(*carried), halt, loss, ce.mean(), q.mean(),
                cell_ok.all(-1), clip)

    def batch_segment(self, s) -> tuple:
        """
        The window of a segment batched, its loss backpropagated with no
        per-board clip: what :meth:`segment` does when ``net.slot_clip``
        is ``None``.

        Parameters:
            s : The state the prefix returned.
        """
        x, *labels = self.targets()
        s, every, halt = self.net.window(s, x)
        loss, ce, q, cell_ok, _ = self.window_loss(every, halt, x, *labels)
        loss.backward()
        return (detach_state(s), halt.detach(), loss.detach(), ce.detach(),
                q.detach(), cell_ok.all(-1), None)

    def iterate(self, optimizer, scheduler, ema) -> dict:
        """
        One segment and one optimizer step: the global norm clip, the step
        rejected when the pre-clip norm exceeds ``skip_spike``, the slow
        reference moved on accepted steps (or set on the first), the
        schedule and the average stepped always, then the halted slots
        refilled. Returns the step's statistics.

        Parameters:
            optimizer : The optimizer of the net.
            scheduler : Its learning-rate schedule.
            ema : The weight average, with ``update(net)``.
        """
        # pylint: disable=too-many-locals
        cfg, net = self.cfg, self.net
        optimizer.zero_grad(set_to_none=True)
        state, halt, loss, ce, q, correct, clip = self.segment()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                   cfg.grad_clip)
        spiked = cfg.skip_spike is not None \
            and float(grad_norm) > cfg.skip_spike
        if spiked:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
        if clip is not None and cfg.slot_ref_decay \
                and (net.slot_ref_running is None or not spiked):
            net.slot_ref_running = update_reference(
                net.slot_ref_running, clip["median"], cfg.slot_ref_decay)
        scheduler.step()
        ema.update(net)
        with torch.no_grad():
            depth = self.depth + 1
            logit = halt_logit(halt.detach().float())
            wants = logit > cfg.halt_threshold
            capped = depth >= cfg.n_sup
            halted = wants | capped
            stats = {"loss": loss.detach(), "ce": ce.detach(), "q": q.detach(),
                     "grad": grad_norm.detach(), "logit": logit.mean(),
                     "spike": float(spiked), "halted": halted.sum(),
                     "solved": (halted & correct).sum(),
                     "depth_sum": (depth * halted).sum(),
                     "capped": (capped & ~wants).sum()}
            self.refill(halted, detach_state(state), depth)
        stats["clip"] = clip
        return stats

    @torch.no_grad()
    def refill(self, halted, state, depth) -> None:
        """
        Carry ``state`` and ``depth`` on, with fresh puzzles, a fresh state
        and depth zero on the ``halted`` slots.

        Parameters:
            halted : Which slots halted ``(rows, )``.
            state : The state the segment ended on.
            depth : The segments each slot has run.
        """
        n = int(halted.sum())
        self.state = state
        self.depth = torch.where(halted, torch.zeros_like(depth), depth)
        if not n:
            return
        slots = tuple(old.clone() for old in self.slots())
        for old, new in zip(slots, self.stream.take(n)):
            old[halted] = new
        self.load(*slots)
        self.state = where_state(halted, self.net.initial(self.x), state)


@torch.no_grad()
def rollout(net, x, target, given, caps, sigma: float = 0.0,
            generator=None, threshold: float = 0.0) -> dict:
    """
    One trajectory to ``max(caps)`` segments from ``net.initial(x)``, the
    answer trace kicked by ``sigma * randn`` before every segment. At each
    cap: ``fixed`` (whether the board right after that many segments is
    correct) and its ``cell`` accuracy, ``depth`` (the segment at which the
    halt logit first cleared ``threshold``, else the cap), and ``best`` /
    ``best_q`` -- whether the board of the most confident segment so far
    is correct, and its halt logit, which best-of-k ranks rollouts by.

    Parameters:
        net : The net, with ``initial(x)`` and ``act_step(s, x)``.
        x : The puzzle lattices ``(rows, cells, channels)``.
        target : The solutions ``(rows, cells)``.
        given : Which cells are given ``(rows, cells)``.
        caps : The segment counts to read the trajectory at.
        sigma : The scale of the noise on the answer trace.
        generator : The generator of the noise, on the device of ``x``.
        threshold : The board halt logit that counts as halting.

    Example
    -------
    >>> from collections import namedtuple  # doctest: +EXTRA
    >>> State = namedtuple("State", "answer")
    >>> class Net:
    ...     def initial(self, x):
    ...         return State(torch.zeros_like(x))
    ...     def act_step(self, s, x):
    ...         s = State(s.answer + x)
    ...         return s, (s.answer, ), s.answer.sum(-1) - 1.5
    >>> x = torch.tensor([[[1., 0.], [0., 1.]]])
    >>> out = rollout(Net(), x, torch.tensor([[0, 1]]), x.sum(-1) == 1, (1, 3))
    >>> out[3]["depth"], out[3]["fixed"], out[3]["best_q"]
    (tensor([3.]), tensor([True]), tensor([0.8069]))
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    rows, device = len(x), x.device
    state = net.initial(x)
    halted = torch.zeros(rows, dtype=torch.bool, device=device)
    step = torch.zeros(rows, dtype=torch.long, device=device)
    best_q = torch.full((rows, ), -float("inf"), device=device)
    best = torch.zeros(rows, dtype=torch.bool, device=device)
    caps, out = set(caps), {}
    for t in range(1, max(caps) + 1):
        if sigma:
            answer = state.answer
            state = state._replace(answer=answer + sigma * torch.randn(
                answer.shape, generator=generator, device=device,
                dtype=answer.dtype))
        state, every, halt = net.act_step(state, x)[:3]
        q = halt_logit(halt.float())
        matches = decode(every[-1], given, x) == target
        correct = matches.all(-1)
        newly = ~halted & (q > threshold)
        step = torch.where(newly, torch.full_like(step, t), step)
        halted = halted | newly
        better = q > best_q
        best_q = torch.where(better, q, best_q)
        best = torch.where(better, correct, best)
        if t in caps:
            out[t] = {"fixed": correct, "cell": matches.float().mean(-1),
                      "depth": torch.where(halted, step,
                                           torch.full_like(step, t)).float(),
                      "best": best.clone(), "best_q": best_q.clone()}
    return out
