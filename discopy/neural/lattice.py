# -*- coding: utf-8 -*-

"""
Lattice deduction: a recursive network that carries a lattice ``x`` in
``{0, 1}^(B, S, C)`` of candidate sets per cell (a blank is all ones, a
given is one-hot) and reads three heads off its answer trace: a keep/kill
logit per candidate (``bce``), a symbol logit per cell (``sm``) and a
conflict logit per cell (``conf``, aggregated to a board logit by
``logsumexp``).

* :func:`project` -- the DPLL-style projection: threshold deduction, the
  board status, one sampled decision per undecided board.
* :func:`losses` -- the deep-supervised loss.
* :class:`Pool` -- the trainer: a persistent pool of lattices, one forward
  per step feeding the loss and the projection, finished or aged entries
  refilled from the data stream.
* :func:`solve` -- the stochastic chain search that scores the model:
  ``n_chains`` chains per puzzle, a chain in conflict reset to the puzzle,
  the first chain of a puzzle to reach ``solved`` accepted.

The trainer reads its net through ``run(x) -> (state, heads, every)``, the
search through ``step(x, given, generator, theta, theta_cls, temp)``, one
supervision step plus :func:`project`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Heads
    Projection
    Config
    Pool
    SolveConfig

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        lattice_of
        conflicts
        board_logit
        decide
        project
        weighted_bce
        losses
        exact
        solve
        evaluate
        describe

Example
-------
>>> x = lattice_of([[0, 2, 0]], vocab=2)  # doctest: +EXTRA
>>> x
tensor([[[1., 1.],
         [0., 1.],
         [1., 1.]]])
>>> heads = Heads(bce=torch.tensor([[[-9., 9.], [0., 0.], [9., 9.]]]),
...               sm=torch.zeros(1, 3, 2), conf=torch.full((1, 3, 1), -9.))
>>> found = project(x, x.sum(-1) == 1, heads, temp=0.)
>>> found.x
tensor([[[0., 1.],
         [0., 1.],
         [1., 0.]]])
>>> found.solved, found.decided
(tensor([False]), tensor([True]))
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch.nn import functional as F

THETA = 0.1
""" The kill threshold of the projection, on ``sigmoid(bce)``. """

THETA_CLS = 0.5
""" The board conflict probability above which training rejects a board. """

TEMP = 1.5
""" The softmax temperature of the decision. """

W_POS, W_NEG = 4.0, 0.5
""" The positive and negative multipliers of the weighted BCE. """

W_SM = 0.2
""" The weight of the symbol cross-entropy in the loss. """

W_CONF = 0.1
""" The weight of the board conflict BCE in the loss. """

W_CELL = 0.1
""" The weight of the per-cell conflict BCE in the loss, when it is on. """


def lattice_of(puzzles, vocab: int = 9) -> torch.Tensor:
    """
    The lattice of digit arrays ``(rows, S)``, ``0`` blank, ``1..vocab``.

    Parameters:
        puzzles : The digits, ``0`` for a blank.
        vocab : The number of symbols.

    Example
    -------
    >>> lattice_of([[0, 1]], vocab=2)  # doctest: +EXTRA
    tensor([[[1., 1.],
             [1., 0.]]])
    """
    puzzles = torch.as_tensor(puzzles, dtype=torch.long)
    hot = F.one_hot(puzzles.clamp(min=1) - 1, vocab).float()
    return torch.where(puzzles.unsqueeze(-1) == 0, torch.ones_like(hot), hot)


def conflicts(x, y) -> tuple:
    """
    ``(cell, board)``: per cell, whether no alive candidate is the
    solution's; per board, whether some cell is.

    Parameters:
        x : The lattices ``(B, S, C)``.
        y : The solution lattices ``(B, S, C)``.

    Example
    -------
    >>> x = torch.tensor([[[1., 0.], [1., 1.]]])  # doctest: +EXTRA
    >>> conflicts(x, torch.tensor([[[0., 1.], [0., 1.]]]))
    (tensor([[ True, False]]), tensor([True]))
    """
    cell = ~((x > 0.5) & (y > 0.5)).any(-1)
    return cell, cell.any(-1)


class Heads(NamedTuple):
    """
    What the three heads read off the answer trace, ``(B, S, .)`` each.

    Parameters:
        bce : The keep/kill logit of every candidate.
        sm : The symbol logits of every cell.
        conf : The conflict logit of every cell.

    Example
    -------
    >>> zeros = torch.zeros(3, 1, 2, 2, requires_grad=True)  # doctest: +EXTRA
    >>> heads = Heads(*zeros)
    >>> heads.detach().sm.requires_grad
    False
    """
    bce: torch.Tensor
    sm: torch.Tensor
    conf: torch.Tensor

    def detach(self) -> Heads:
        """ The heads, detached. """
        return Heads(self.bce.detach(), self.sm.detach(), self.conf.detach())


def board_logit(conf) -> torch.Tensor:
    """
    The board conflict logit, ``logsumexp`` of the per-cell logits.

    Parameters:
        conf : The per-cell conflict logits ``(..., S, 1)``.

    Example
    -------
    >>> board_logit(torch.zeros(1, 2, 1))  # doctest: +EXTRA
    tensor([0.6931])
    """
    return torch.logsumexp(conf.squeeze(-1), dim=-1)


class Projection(NamedTuple):
    """
    What :func:`project` did to a lattice.

    Parameters:
        x : The projected lattice.
        conflict : ``(B, )`` bool, an empty cell or the conflict head.
        solved : ``(B, )`` bool, all singletons and no conflict.
        kill : ``(B, S, C)`` bool, the deduced bits.
        decided : ``(B, )`` bool, the rows the decision pinned.

    Example
    -------
    >>> Projection._fields
    ('x', 'conflict', 'solved', 'kill', 'decided')
    """
    x: torch.Tensor
    conflict: torch.Tensor
    solved: torch.Tensor
    kill: torch.Tensor
    decided: torch.Tensor


@torch.no_grad()
def decide(x, sm, decided, temp: float = TEMP, generator=None):
    """
    The decision at a fixed shape: a cell and a symbol are sampled for
    every row and written, in place, only where ``decided`` holds, so that
    no row index travels to the host. The cell is uniform over the cells
    with several candidates (over all cells when there is none), the symbol
    is drawn from the candidate-masked softmax of ``sm`` at ``temp``, or is
    its argmax when ``temp`` is zero.

    Parameters:
        x : The lattices ``(B, S, C)``, written in place.
        sm : The symbol logits ``(B, S, C)``.
        decided : Which rows to pin ``(B, )``.
        temp : The softmax temperature.
        generator : The generator of the draws.

    Example
    -------
    >>> x = torch.tensor([[[1., 1.], [0., 1.]]])  # doctest: +EXTRA
    >>> sm = torch.tensor([[[0., 1.], [0., 0.]]])
    >>> decide(x, sm, torch.tensor([True]), temp=0.)
    tensor([[[0., 1.],
             [0., 1.]]])
    """
    rows = torch.arange(len(x), device=x.device)
    multi = (x.sum(-1) > 1.5).double()
    multi = multi + (multi.sum(-1, keepdim=True) == 0)
    cell = torch.multinomial(multi, 1, generator=generator).squeeze(-1)
    alive_at = x[rows, cell] > 0.5
    logits = sm[rows, cell].masked_fill(~alive_at, -torch.inf)
    logits = torch.where(alive_at.any(-1, keepdim=True), logits, 0.0)
    if temp > 0:
        probs = torch.softmax(logits / temp, dim=-1)
        digit = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    else:
        digit = logits.argmax(-1)
    pin = F.one_hot(digit, x.shape[-1]).to(x.dtype)
    x[rows, cell] = torch.where(decided.unsqueeze(-1), pin, x[rows, cell])
    return x


@torch.no_grad()
def project(x, given, heads: Heads, theta: float = THETA,
            theta_cls: float = THETA_CLS, temp: float = TEMP,
            generator=None, fixed_shape: bool = False) -> Projection:
    """
    Kill the candidates under ``theta`` (never on a given), read the status,
    then on every undecided board pin one uniform multi-candidate cell to a
    symbol sampled from its candidate-masked softmax at ``temp``.

    ``fixed_shape`` decides through :func:`decide`, drawing for every row;
    otherwise only the decided rows are gathered and drawn for, so that the
    draws depend on how many there are.

    Parameters:
        x : The lattices ``(B, S, C)``.
        given : Which cells are given ``(B, S)``, never killed.
        heads : The heads read off the answer trace.
        theta : The kill threshold on ``sigmoid(bce)``.
        theta_cls : The threshold on the board conflict probability.
        temp : The softmax temperature of the decision.
        generator : The generator of the draws.
        fixed_shape : Whether to draw for every row.

    Example
    -------
    >>> x = torch.ones(2, 2, 2)  # doctest: +EXTRA
    >>> heads = Heads(torch.zeros(2, 2, 2), torch.zeros(2, 2, 2),
    ...               torch.tensor([[[-9.], [-9.]], [[9.], [9.]]]))
    >>> found = project(x, x.sum(-1) == 1, heads, fixed_shape=True,
    ...                 generator=torch.Generator().manual_seed(0))
    >>> found.conflict, found.decided, found.x.sum((1, 2))
    (tensor([False,  True]), tensor([ True, False]), tensor([3., 4.]))
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    kill = (torch.sigmoid(heads.bce) < theta) & (x > 0.5) \
        & ~given.unsqueeze(-1)
    x = x.masked_fill(kill, 0.0)
    alive = x.sum(-1)
    fires = torch.sigmoid(board_logit(heads.conf)) > theta_cls
    conflict = (alive == 0).any(-1) | fires
    solved = (alive == 1).all(-1) & ~conflict
    decided = ~conflict & ~solved
    if fixed_shape:
        x = decide(x, heads.sm, decided, temp, generator)
        return Projection(x, conflict, solved, kill, decided)
    if decided.any():
        rows = decided.nonzero(as_tuple=True)[0]
        multi = (x[rows].sum(-1) > 1.5).double()
        cell = torch.multinomial(multi, 1, generator=generator).squeeze(-1)
        alive_at = x[rows, cell] > 0.5
        logits = heads.sm[rows, cell].masked_fill(~alive_at, -torch.inf)
        if temp > 0:
            probs = torch.softmax(logits / temp, dim=-1)
            digit = torch.multinomial(
                probs, 1, generator=generator).squeeze(-1)
        else:
            digit = logits.argmax(-1)
        x[rows, cell] = 0.0
        x[rows, cell, digit] = 1.0
    return Projection(x, conflict, solved, kill, decided)


def weighted_bce(logits, target, w_pos: float = W_POS,
                 w_neg: float = W_NEG):
    """
    The binary cross-entropy with positive and negative multipliers,
    mean-reduced.

    Parameters:
        logits : The logits.
        target : The labels in ``[0, 1]``.
        w_pos : The multiplier of the positive term.
        w_neg : The multiplier of the negative term.

    Example
    -------
    >>> weighted_bce(torch.zeros(2), torch.tensor([1., 0.]))  # doctest: +EXTRA
    tensor(1.5596)
    """
    pos = target * F.logsigmoid(logits)
    neg = (1.0 - target) * F.logsigmoid(-logits)
    return -(w_pos * pos + w_neg * neg).mean()


def losses(every, x, target, given, cell_conf: bool = False) -> tuple:
    """
    The loss averaged over every readout of a deep-supervised step, on the
    pre-projection lattice ``x`` against the (possibly multi-hot)
    ``target``: the weighted BCE against ``x * target``, :data:`W_SM` x the
    symbol cross-entropy on the cells that are neither given, nor
    multi-alive in the target, nor on a board with no solution left,
    :data:`W_CONF` x the board conflict BCE and, under ``cell_conf``,
    :data:`W_CELL` x the per-cell conflict BCE. Returns the total and the
    detached parts.

    Parameters:
        every : The :class:`Heads` of every differentiated cycle.
        x : The lattices ``(B, S, C)`` before the projection.
        target : The solution lattices ``(B, S, C)``.
        given : Which cells are given ``(B, S)``.
        cell_conf : Whether to add the per-cell conflict term.

    Example
    -------
    >>> x = torch.ones(1, 2, 2)  # doctest: +EXTRA
    >>> y = torch.tensor([[[1., 0.], [0., 1.]]])
    >>> every = [Heads(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2),
    ...                torch.zeros(1, 2, 1))]
    >>> total, parts = losses(every, x, y, x.sum(-1) == 1, cell_conf=True)
    >>> sorted(parts), round(float(total), 4)
    (['bce', 'cell_conf', 'conf', 'sm'], 1.8774)
    """
    # pylint: disable=too-many-locals
    n = len(every)
    heads = Heads(*(torch.stack(head) for head in zip(*every)))
    cell, board = conflicts(x, target)
    total = weighted_bce(heads.bce, (x * target).unsqueeze(0).expand_as(
        heads.bce))
    parts = {"bce": total.detach()}
    mask = (~given & ~(target.sum(-1) > 1.5)
            & ~board.unsqueeze(-1)).to(heads.sm.dtype)
    weight = mask.unsqueeze(0).expand(n, -1, -1).reshape(-1)
    digits = F.cross_entropy(
        heads.sm.reshape(-1, heads.sm.shape[-1]),
        target.argmax(-1).unsqueeze(0).expand(n, -1, -1).reshape(-1),
        reduction="none")
    digits = (digits * weight).sum() / weight.sum().clamp(min=1.0)
    total = total + W_SM * digits
    parts["sm"] = digits.detach()
    conf = F.binary_cross_entropy_with_logits(
        board_logit(heads.conf),
        board.to(heads.conf.dtype).unsqueeze(0).expand(n, -1))
    total = total + W_CONF * conf
    parts["conf"] = conf.detach()
    if cell_conf:
        extra = F.binary_cross_entropy_with_logits(
            heads.conf.squeeze(-1),
            cell.to(heads.conf.dtype).unsqueeze(0).expand(n, -1, -1))
        total = total + W_CELL * extra
        parts["cell_conf"] = extra.detach()
    return total, parts


@dataclass
class Config:
    """
    The lattice training protocol.

    Parameters:
        steps : The optimizer steps.
        batch_size : The pool size, which is the batch size.
        lr : The peak learning rate.
        weight_decay : The decay of the matrices.
        warmup_frac : The fraction of the steps the warmup takes.
        betas : The moment decays of AdamW.
        max_age : The steps after which an entry is refilled.
        ema_decay : The decay of the weight average.
        grad_clip : The global norm clip.
        theta : The kill threshold of the projection.
        theta_cls : Its threshold on the board conflict probability.
        temp : Its softmax temperature.
        seed : The seed a trainer draws the projection's generator from,
               see :meth:`Pool.iterate`.
        compile : Whether to compile the loss.
        compile_mode : The mode it is compiled in, ``None`` for the default.
        amp_dtype : The CUDA autocast dtype, ``None`` for no autocast.
        cell_conf : Whether the loss has the per-cell conflict term.

    Example
    -------
    >>> Config(steps=10).compile_mode
    'reduce-overhead'
    """
    # pylint: disable=too-many-instance-attributes  # one per knob
    steps: int = 20000
    batch_size: int = 512
    lr: float = 4.8119e-3
    weight_decay: float = 0.04821
    warmup_frac: float = 0.1009
    betas: tuple = (0.9, 0.95)
    max_age: int = 100
    ema_decay: float = 0.9995
    grad_clip: float = 0.5
    theta: float = THETA
    theta_cls: float = THETA_CLS
    temp: float = TEMP
    seed: int = 0
    compile: bool = True
    compile_mode: str | None = "reduce-overhead"
    amp_dtype: str | None = "bfloat16"
    cell_conf: bool = False


class Pool:
    """
    The persistent pool, which is the batch: per entry the current lattice
    ``x``, its start ``x0`` (whose singletons are the protected givens), its
    labels and an age. Every :meth:`iterate` runs one forward on the whole
    pool, feeding the deep loss (with a gradient) and the fixed-shape
    projection (detached), then the optimizer, the schedule and the
    average. The entries :meth:`verdict` discards are replaced from the
    stream at age zero.

    The labels are what the stream's ``take(n)`` returns after the start
    lattices: here one solution lattice ``y`` per entry, the loss
    :meth:`target`; an entry is discarded when it is solved and correct,
    when its detected conflict is real, or when it reaches ``max_age``.

    Parameters:
        net : The recursive network, with ``run(x)``.
        stream : Where fresh puzzles come from, by ``take(n)``.
        cfg : The protocol.

    Note
    ----
    A variant of the protocol overrides a method, never the op order:
    :meth:`label` and :meth:`relabel` (what the labels are),
    :meth:`target` (what the loss is against) and :meth:`verdict` (which
    entries are finished, and what is counted).

    Example
    -------
    >>> class Net(torch.nn.Module):  # doctest: +EXTRA
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.bce = torch.nn.Linear(2, 2)
    ...         self.sm = torch.nn.Linear(2, 2)
    ...         self.conf = torch.nn.Linear(2, 1)
    ...     def run(self, x):
    ...         heads = Heads(self.bce(x), self.sm(x), self.conf(x))
    ...         return None, heads, (heads, )
    >>> class Stream:
    ...     def take(self, n):
    ...         y = torch.eye(2)[[0, 1, 0]].expand(n, 3, 2)
    ...         return torch.ones(n, 3, 2), y
    >>> _ = torch.manual_seed(0)
    >>> net = Net()
    >>> pool = Pool(net, Stream(), Config(batch_size=4, compile=False))
    >>> optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    >>> scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    >>> stats = pool.iterate(optimizer, scheduler,
    ...                      generator=torch.Generator().manual_seed(0))
    >>> list(stats)  # doctest: +NORMALIZE_WHITESPACE
    ['solved_correct', 'tp_conflict', 'aged', 'discarded', 'loss',
     'bce', 'sm', 'conf']
    """
    # pylint: disable=too-many-instance-attributes  # one per pool field
    COUNTERS = ("solved_correct", "tp_conflict", "aged", "discarded")

    def __init__(self, net, stream, cfg: Config):
        self.net, self.stream, self.cfg = net, stream, cfg
        x0, *labels = stream.take(cfg.batch_size)
        self.label(*labels)
        self.x, self.x0 = x0.clone(), x0
        self.age = torch.zeros(cfg.batch_size, dtype=torch.long,
                               device=x0.device)
        mode = {"mode": cfg.compile_mode} if cfg.compile_mode else {}
        self.losses = torch.compile(losses, **mode) if cfg.compile else losses
        device = next(net.parameters()).device
        self.amp_dtype = getattr(torch, cfg.amp_dtype) \
            if cfg.amp_dtype is not None and device.type == "cuda" else None

    def label(self, y) -> None:
        """
        Hold the labels of the first entries.

        Parameters:
            y : The solution lattices ``(B, S, C)``.
        """
        self.y = y

    def relabel(self, discard, y) -> None:
        """
        Replace the labels of the ``discard`` entries.

        Parameters:
            discard : Which entries are replaced ``(B, )``.
            y : Their new solution lattices.
        """
        self.y = self.y.clone()
        self.y[discard] = y

    def target(self) -> torch.Tensor:
        """ What the loss of this step is against: the solutions ``y``. """
        return self.y

    def verdict(self, found: Projection) -> tuple:
        """
        Which entries are finished after the projection ``found``, and the
        counters: ``(discard, {name: count})``, ``discarded`` among them.

        Parameters:
            found : What :func:`project` did to the pool.
        """
        board_ok = (found.x.argmax(-1) == self.y.argmax(-1)).all(-1)
        solved_correct = found.solved & board_ok
        tp_conflict = found.conflict & conflicts(found.x, self.y)[1]
        aged = self.age >= self.cfg.max_age
        discard = solved_correct | tp_conflict | aged
        return discard, {"solved_correct": solved_correct.sum(),
                         "tp_conflict": tp_conflict.sum(), "aged": aged.sum(),
                         "discarded": discard.sum()}

    def amp(self):
        """ The CUDA autocast of the forward, or a null context. """
        return torch.autocast("cuda", dtype=self.amp_dtype) \
            if self.amp_dtype is not None else contextlib.nullcontext()

    def iterate(self, optimizer, scheduler, ema=None, generator=None) -> dict:
        """
        One optimizer step on the whole pool, then discard and refill.
        Returns the counters, the loss and its parts.

        Parameters:
            optimizer : The optimizer of the net.
            scheduler : Its learning-rate schedule.
            ema : The weight average, with ``update(net)``, or ``None``.
            generator : The generator of the projection's draws, ``None`` for
                        the global one: the pool does not seed it.
        """
        cfg = self.cfg
        given = self.x0.sum(-1) == 1
        target = self.target()
        torch.compiler.cudagraph_mark_step_begin()
        with self.amp():
            _, heads, every = self.net.run(self.x)
            loss, parts = self.losses(
                every or (heads, ), self.x, target, given, cfg.cell_conf)
        found = project(self.x, given, heads.detach(), cfg.theta,
                        cfg.theta_cls, cfg.temp, generator, fixed_shape=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(self.net)
        stats = self.refill(found)
        stats["loss"] = loss.detach()
        stats.update(parts)
        return stats

    @torch.no_grad()
    def refill(self, found: Projection) -> dict:
        """
        Age every entry, keep the projected lattices, replace the finished
        entries at age zero. Returns the counters, read in one host
        transfer.

        Parameters:
            found : What :func:`project` did to the pool.
        """
        self.age += 1
        discard, counted = self.verdict(found)
        fresh = found.x.detach().clone()
        stats = dict(zip(
            counted, torch.stack(list(counted.values())).tolist()))
        n = stats["discarded"]
        if n:
            x_new, *labels = self.stream.take(n)
            fresh[discard] = x_new
            self.x0 = self.x0.clone()
            self.x0[discard] = x_new
            self.relabel(discard, *labels)
            self.age[discard] = 0
        self.x = fresh
        return stats


@dataclass
class SolveConfig:
    """
    The chain search protocol.

    Parameters:
        max_rounds : The per-puzzle budget of model calls.
        n_chains : The chains per puzzle.
        batch_size : The rows per model call.
        theta : The kill threshold of the projection.
        theta_cls : Its threshold on the board conflict probability.
        temp : Its softmax temperature.
        sync_every : The accept/reset cadence, in model calls.
        seed : The seed of the search's generator.
        amp_dtype : The CUDA autocast dtype, ``None`` for no autocast.

    Example
    -------
    >>> SolveConfig().batch_size // SolveConfig().n_chains
    64
    """
    # pylint: disable=too-many-instance-attributes  # one per knob
    max_rounds: int = 1000
    n_chains: int = 64
    batch_size: int = 4096
    theta: float = THETA
    theta_cls: float = 0.8
    temp: float = TEMP
    sync_every: int = 1
    seed: int = 0
    amp_dtype: str | None = None


def exact(board, truth) -> str:
    """
    The bucket of an accepted board: ``correct`` when it is the solution,
    ``wrong`` otherwise.

    Parameters:
        board : The accepted symbols ``(S, )``.
        truth : The solution's ``(S, )``.

    Example
    -------
    >>> exact(torch.tensor([1, 2]), torch.tensor([1, 2]))  # doctest: +EXTRA
    'correct'
    """
    return "correct" if bool((board == truth).all()) else "wrong"


@torch.no_grad()
def solve(model, puzzles, solutions, cfg: SolveConfig, grade=exact,
          prefer=None, label=None) -> dict:
    """
    The stochastic chain search on lattices ``puzzles`` ``(P, S, C)``.

    ``M = batch_size // n_chains`` slots each hold one puzzle's
    ``n_chains`` chains, and the whole batch is stepped every round by
    ``model.step``. A chain in conflict is reset to the puzzle; the first
    chain of a slot to reach ``solved`` is accepted -- the model's own
    signal, ``grade`` judges it against ``solutions`` afterwards -- and the
    slot is refilled; a slot whose budget of model calls runs out is a
    timeout. Returns per puzzle its ``bucket``, ``round_solved`` and
    ``puzzle_calls``, and the total ``model_calls``.

    Parameters:
        model : The net, with ``step(x, given, generator, theta,
            theta_cls, temp)``.
        puzzles : The start lattices ``(P, S, C)``.
        solutions : The solution lattices ``(P, S, C)``.
        cfg : The protocol.
        grade : The bucket of an accepted board, from its symbols and the
            solution's.
        prefer : Which finished chains to accept first, from the lattices
            and the chains' labels, or ``None``.
        label : One value per puzzle ``(n, )``, copied onto its chains, for
                ``prefer``.

    Example
    -------
    >>> class Model:  # doctest: +EXTRA
    ...     def step(self, x, given, generator, theta, theta_cls, temp):
    ...         sm = torch.tensor([0., 9.]).expand_as(x)
    ...         heads = Heads(x - .5, sm, x[..., :1] - 9)
    ...         return project(x, given, heads, theta, theta_cls, temp,
    ...                        generator)
    >>> puzzles = lattice_of([[1, 0], [0, 1]], vocab=2)
    >>> out = solve(Model(), puzzles, lattice_of([[1, 2], [2, 1]], vocab=2),
    ...             SolveConfig(max_rounds=4, n_chains=2, batch_size=2))
    >>> out["bucket"], out["model_calls"], out["puzzle_calls"].tolist()
    (['correct', 'correct'], 4, [2, 2])
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements
    truth = solutions.argmax(-1).cpu()
    P, S, C = puzzles.shape
    K = cfg.n_chains
    M = max(1, cfg.batch_size // K)
    device = puzzles.device
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    given_all = puzzles.sum(-1) == 1

    x = torch.zeros(M * K, S, C, device=device, dtype=puzzles.dtype)
    x0 = torch.zeros_like(x)
    given = torch.zeros(M * K, S, dtype=torch.bool, device=device)
    chain_done = torch.ones(M * K, dtype=torch.bool, device=device)
    chain_label = None if label is None else torch.ones(
        M * K, dtype=label.dtype, device=device)
    slot_puzzle = torch.full((M, ), -1, dtype=torch.long)
    slot_round = torch.zeros(M, dtype=torch.long)
    slot_start = torch.zeros(M, dtype=torch.long)
    bucket = ["timeout"] * P
    round_solved = torch.full((P, ), -1, dtype=torch.long)
    puzzle_calls = torch.full((P, ), -1, dtype=torch.long)
    total_calls, next_puzzle = 0, 0
    amp_dtype = getattr(torch, cfg.amp_dtype) \
        if cfg.amp_dtype is not None and device.type == "cuda" else None

    def fill(slot: int, p: int) -> None:
        rows = slice(slot * K, (slot + 1) * K)
        x[rows], x0[rows], given[rows] = puzzles[p], puzzles[p], given_all[p]
        chain_done[rows] = False
        if label is not None:
            chain_label[rows] = label[p]  # ty: ignore[invalid-assignment]
        slot_puzzle[slot], slot_round[slot] = p, 0
        slot_start[slot] = total_calls

    def accept(solved, conflict, preferred, x_new) -> list:
        evictions = []
        for slot in range(M):
            p = int(slot_puzzle[slot])
            if p < 0:
                continue
            lo, hi = slot * K, (slot + 1) * K
            if bool(solved[lo:hi].any()):
                pick = preferred[lo:hi] if preferred is not None \
                    and bool(preferred[lo:hi].any()) else solved[lo:hi]
                b = lo + int(pick.nonzero()[0, 0])
                bucket[p] = grade(x_new[b].argmax(-1).cpu(), truth[p])
                round_solved[p] = slot_round[slot]
                evictions.append(slot)
                continue
            in_conflict = conflict[lo:hi]
            if bool(in_conflict.any()):
                rows = in_conflict.nonzero()[:, 0].to(device) + lo
                x_new[rows] = x0[rows]
        return evictions

    for slot in range(min(M, P)):
        fill(slot, next_puzzle)
        next_puzzle += 1

    while bool((slot_puzzle >= 0).any()):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            step = model.step(x, given, generator, cfg.theta, cfg.theta_cls,
                              cfg.temp)
        total_calls += 1
        active = ~chain_done
        x_new = torch.where(chain_done.view(-1, 1, 1), x, step.x)
        evictions = []
        if total_calls % cfg.sync_every == 0:
            done = step.solved & active
            preferred = None if prefer is None \
                else (done & prefer(x_new, chain_label)).cpu()
            conflict = (step.conflict & active).cpu()
            evictions = accept(done.cpu(), conflict, preferred, x_new)
        for slot in range(M):
            p = int(slot_puzzle[slot])
            if p < 0 or slot in evictions:
                continue
            slot_round[slot] += 1
            if int(slot_round[slot]) >= cfg.max_rounds:
                evictions.append(slot)
        x = x_new
        for slot in sorted(evictions):
            p = int(slot_puzzle[slot])
            puzzle_calls[p] = total_calls - slot_start[slot]
            slot_puzzle[slot] = -1
            chain_done[slot * K:(slot + 1) * K] = True
            if next_puzzle < P:
                fill(slot, next_puzzle)
                next_puzzle += 1

    return {"bucket": bucket, "round_solved": round_solved,
            "puzzle_calls": puzzle_calls, "model_calls": total_calls}


def evaluate(net, x, y, cfg: SolveConfig) -> dict:
    """
    The chain search's summary on lattices ``x``, with its seconds.

    Parameters:
        net : The net, with ``step``.
        x : The start lattices ``(P, S, C)``.
        y : The solution lattices ``(P, S, C)``.
        cfg : The protocol.

    Example
    -------
    >>> class Model:  # doctest: +EXTRA
    ...     def step(self, x, given, generator, theta, theta_cls, temp):
    ...         heads = Heads(x - .5, torch.zeros_like(x), x[..., :1] - 9)
    ...         return project(x, given, heads, theta, theta_cls, temp,
    ...                        generator)
    >>> y = lattice_of([[1, 2]], vocab=2)
    >>> score = evaluate(Model(), y, y, SolveConfig(n_chains=2, batch_size=2))
    >>> score["correct"], score["board"], score["calls_p50"]
    (1, 1.0, 1.0)
    """
    tick = time.perf_counter()
    out = solve(net, x, y, cfg)
    counts = {name: out["bucket"].count(name)
              for name in ("correct", "wrong", "timeout")}
    calls = out["puzzle_calls"][out["puzzle_calls"] >= 0].float()
    rounds = out["round_solved"][out["round_solved"] >= 0].float()
    n = len(out["bucket"])
    return {"n": n, **counts, "board": counts["correct"] / n,
            "model_calls": out["model_calls"],
            **({f"calls_p{int(100 * q)}": float(calls.quantile(q))
                for q in (0.5, 0.9)} if len(calls) else {}),
            "rounds_p50": float(rounds.quantile(0.5)) if len(rounds) else -1.0,
            "seconds": time.perf_counter() - tick}


def describe(name: str, score: dict) -> str:
    """
    One line of :func:`evaluate`'s summary.

    Parameters:
        name : What was scored.
        score : The summary.

    Example
    -------
    >>> describe("valid", {"correct": 497, "n": 500, "board": .994,
    ...                    "wrong": 0, "timeout": 3, "calls_p50": 12.,
    ...                    "seconds": 30.})
    'valid: 497/500 (99.4%), 0 wrong, 3 timeout, median 12 calls (30s)'
    """
    return (f"{name}: {score['correct']}/{score['n']} ({score['board']:.1%}),"
            f" {score['wrong']} wrong, {score['timeout']} timeout, median "
            f"{score.get('calls_p50', float('nan')):.0f} calls "
            f"({score['seconds']:.0f}s)")
