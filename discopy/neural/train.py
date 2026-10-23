# -*- coding: utf-8 -*-

"""
Optimisation for the recursive networks of :mod:`discopy.neural`: seeding,
AdamW decaying the matrices only, a linear warmup then a cosine decay to a
floor, a bias-corrected weight average and the bounded per-board influence
of a gradient.

The per-board clip (``slot_clip``) guards a deep recurrent segment against
the heavy tail of per-example gradients. Board ``b``'s contribution ``g_b``
to the gradient is scaled by ``min(1, k * reference / r_b)`` before the
batch mean, with ``r_b = |g_b| / loss_b`` its gradient norm per unit of its
own loss; a board whose ``r_b`` is not finite contributes nothing.
``reference`` is a slow running value of the batch median of ``r``
(:func:`update_reference`), or the batch's own median when ``None``. The
same arithmetic runs on rows a backward already returns one per board
(:func:`clip_rows`) and on rows ``torch.func.vmap(grad(...))`` computes
(:func:`per_board_backward`).

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    EMA
    Bound

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        seed_everything
        adamw
        cosine_schedule
        positive
        update_reference
        ratios
        scales
        statistics
        rescale
        clip_rows
        per_board_backward

Example
-------
>>> net = torch.nn.Linear(3, 1)  # doctest: +EXTRA
>>> optimizer = adamw(net, lr=1e-2, weight_decay=0.1)
>>> scheduler = cosine_schedule(optimizer, warmup=1, total=4)
>>> ema = EMA(net, decay=0.9)
>>> for _ in range(4):
...     optimizer.zero_grad(set_to_none=True)
...     net(torch.ones(2, 3)).pow(2).mean().backward()
...     _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
...     optimizer.step()
...     scheduler.step()
...     ema.update(net)
>>> ema.t, sorted(ema.weights())
(4, ['bias', 'weight'])
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

MAX_STEP = math.log(4.0)
""" The largest log-space innovation of the slow reference in one step. """


def seed_everything(seed: int) -> None:
    """
    Seed torch, every CUDA device and numpy's global generator.

    Parameters:
        seed : The seed.

    Example
    -------
    >>> seed_everything(0)  # doctest: +EXTRA
    >>> first = torch.rand(())
    >>> seed_everything(0)
    >>> assert torch.equal(first, torch.rand(()))
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def adamw(model, lr: float, weight_decay: float,
          betas=(0.9, 0.95)) -> torch.optim.AdamW:
    """
    AdamW decaying the matrix weights only (the parameters with at least
    two axes), fused when the first parameter is on CUDA.

    Parameters:
        model : The module whose parameters are optimised.
        lr : The learning rate.
        weight_decay : The decay of the matrices.
        betas : The moment decays.

    Example
    -------
    >>> net = torch.nn.Linear(3, 2)  # doctest: +EXTRA
    >>> optimizer = adamw(net, lr=1e-3, weight_decay=0.1)
    >>> [len(group["params"]) for group in optimizer.param_groups]
    [1, 1]
    >>> [group["weight_decay"] for group in optimizer.param_groups]
    [0.1, 0.0]
    """
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": rest, "weight_decay": 0.0}], lr=lr, betas=betas,
        fused=next(model.parameters()).is_cuda)


def cosine_schedule(optimizer, warmup: int, total: int, floor: float = 0.05):
    """
    A linear warmup over ``warmup`` steps, then a cosine decay to ``floor``
    times the peak at step ``total``.

    Parameters:
        optimizer : The optimizer whose learning rate is scheduled.
        warmup : The steps of the linear warmup.
        total : The step at which the decay reaches the floor.
        floor : The final fraction of the peak.

    Example
    -------
    >>> weight = torch.zeros(1, requires_grad=True)  # doctest: +EXTRA
    >>> optimizer = torch.optim.SGD([weight], lr=1.)
    >>> scheduler = cosine_schedule(optimizer, warmup=2, total=6, floor=0.)
    >>> rates = []
    >>> for _ in range(6):
    ...     rates.append(round(scheduler.get_last_lr()[0], 3))
    ...     optimizer.step()
    ...     scheduler.step()
    >>> rates
    [0.5, 1.0, 1.0, 0.854, 0.5, 0.146]
    """
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return floor + (1 - floor) * 0.5 * (
            1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """
    An exponential moving average of the floating-point entries of a
    module's ``state_dict``, bias corrected: after ``t`` updates the
    accumulator still holds ``decay^t`` of the initial weights, and
    :meth:`weights` removes it exactly, ``(shadow - decay^t init) / (1 -
    decay^t)``.

    Parameters:
        model : The module whose weights are averaged.
        decay : The weight the average keeps at every update.

    Example
    -------
    >>> net = torch.nn.Linear(1, 1, bias=False)  # doctest: +EXTRA
    >>> with torch.no_grad():
    ...     _ = net.weight.fill_(1.)
    >>> ema = EMA(net, decay=0.5)
    >>> with torch.no_grad():
    ...     _ = net.weight.fill_(3.)
    >>> ema.update(net)
    >>> ema.shadow["weight"], ema.weights()["weight"]
    (tensor([[2.]]), tensor([[3.]]))
    >>> with torch.no_grad():
    ...     _ = net.weight.fill_(5.)
    >>> with ema.averaged(net):
    ...     net.weight.item()
    3.0
    >>> net.weight.item()
    5.0
    """
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {key: value.detach().clone()
                       for key, value in model.state_dict().items()
                       if value.dtype.is_floating_point}
        self.init = {key: value.detach().clone()
                     for key, value in self.shadow.items()}
        self.t = 0

    @torch.no_grad()
    def update(self, model) -> None:
        """
        Move the average towards the live weights of ``model``, in place:
        ``shadow + (1 - decay) * (live - shadow)``, one ``_foreach_lerp_``
        over every entry, whose rounding the average has.

        Parameters:
            model : The module the average was built from.
        """
        state = model.state_dict()
        torch._foreach_lerp_(  # pylint: disable=protected-access
            [self.shadow[key] for key in self.shadow],
            [state[key] for key in self.shadow], 1.0 - self.decay)
        self.t += 1

    @torch.no_grad()
    def weights(self) -> dict:
        """ The bias-corrected average, computed in float64. """
        bias = float(self.decay) ** int(self.t)
        if self.t <= 0 or not 0.0 <= bias < 1.0:
            return {key: value.detach().clone()
                    for key, value in self.shadow.items()}
        out = {}
        for key, value in self.shadow.items():
            start = self.init[key]
            fixed = value.double() - bias * start.to(value.device).double()
            out[key] = (fixed / (1.0 - bias)).to(value.dtype)
        return out

    @contextlib.contextmanager
    def averaged(self, model):
        """
        Swap the corrected average into ``model``, restore its live weights
        after.

        Parameters:
            model : The module the average was built from.
        """
        backup = {key: value.detach().clone()
                  for key, value in model.state_dict().items()
                  if key in self.shadow}
        model.load_state_dict(self.weights(), strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


def positive(k) -> float | None:
    """
    The clip factor ``k`` as a positive float, or ``None`` (off).

    Parameters:
        k : A positive number, or ``None``.

    Example
    -------
    >>> positive(5), positive(None)
    (5.0, None)
    >>> positive(0)
    Traceback (most recent call last):
    ...
    ValueError: slot_clip must be None or > 0, not 0.0
    """
    if k is None:
        return None
    k = float(k)
    if not k > 0.0:  # pylint: disable=unnecessary-negation  # NaN too
        raise ValueError(f"slot_clip must be None or > 0, not {k}")
    return k


def update_reference(current: float | None, median: float,
                     decay: float) -> float | None:
    """
    The slow reference after an accepted step whose batch median of ``r``
    was ``median``: ``current`` moved in log space by ``(1 - decay)`` times
    the innovation ``log(median / current)``, clamped to :data:`MAX_STEP`
    first. ``None`` takes ``median``; a non-finite or non-positive
    ``median`` leaves ``current`` alone.

    Parameters:
        current : The reference before the step, or ``None``.
        median : The batch median of the ratios ``r``.
        decay : The fraction of the log reference kept at every step.

    Example
    -------
    >>> update_reference(None, 2., decay=0.99)
    2.0
    >>> update_reference(2., 8., decay=0.5)
    4.0
    >>> update_reference(1., 100., decay=0.)
    4.0
    >>> update_reference(1., float("nan"), decay=0.)
    1.0
    """
    median = float(median)
    if not (math.isfinite(median) and median > 0.0):
        return current
    if current is None:
        return median
    innovation = math.log(median / current)
    innovation = max(-MAX_STEP, min(MAX_STEP, innovation))
    return current * math.exp((1.0 - decay) * innovation)


def ratios(norms: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
    """
    The gradient norm of each board per unit of its loss, ``r_b = n_b /
    |loss_b|``.

    Parameters:
        norms : The per-board gradient norms ``(B, )``.
        losses : The per-board losses.

    Example
    -------
    >>> norms = torch.tensor([2., 3.])  # doctest: +EXTRA
    >>> ratios(norms, torch.tensor([1., -.5]))
    tensor([2., 6.])
    """
    losses = losses.detach().to(norms.dtype).reshape(norms.shape)
    return norms / losses.abs().clamp_min(torch.finfo(norms.dtype).tiny)


def scales(norms: torch.Tensor, k: float, losses: torch.Tensor,
           reference: float | None = None, eps: float = 1e-30) -> torch.Tensor:
    """
    The per-board scale ``min(1, k * reference / r_b)``, zero where ``r_b``
    is not finite.

    Parameters:
        norms : The per-board gradient norms ``(B, )``.
        k : The clip factor.
        losses : The per-board losses.
        reference : The slow reference, or ``None`` for the batch median.
        eps : What keeps the division finite.

    Example
    -------
    >>> norms = torch.tensor([1., 2., 8., float("inf")])  # doctest: +EXTRA
    >>> scales(norms, 2., torch.ones(4))
    tensor([1.0000, 1.0000, 0.5000, 0.0000])
    >>> scales(norms, 2., torch.ones(4), reference=1.)
    tensor([1.0000, 1.0000, 0.2500, 0.0000])
    """
    r = ratios(norms, losses)
    finite = torch.isfinite(r)
    cap = k * (r.nanmedian() if reference is None else float(reference))
    scale = (cap / (r + eps)).clamp(max=1.0)
    return torch.where(finite, scale, torch.zeros_like(scale))


def statistics(norms, losses, scale) -> dict:
    """
    The batch median of the ratios ``r`` and the fraction of boards
    clipped.

    Parameters:
        norms : The per-board gradient norms ``(B, )``.
        losses : The per-board losses.
        scale : The scales of :func:`scales`.

    Example
    -------
    >>> norms = torch.tensor([1., 2., 8.])  # doctest: +EXTRA
    >>> statistics(norms, torch.ones(3), torch.tensor([1., 1., .5]))
    {'median': 2.0, 'clipped_frac': 0.3333333333333333}
    """
    return {"median": float(ratios(norms, losses).nanmedian()),
            "clipped_frac": float((scale < 1.0).to(torch.float64).mean())}


def rescale(row: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    The rows ``(B, ...)`` scaled per board, a zero scale zeroing its row
    even where it is not finite.

    Parameters:
        row : The per-board rows, the board first.
        scale : The scale of each board ``(B, )``.

    Example
    -------
    >>> row = torch.tensor([[1., 2.], [float("nan"), 1.]])  # doctest: +EXTRA
    >>> rescale(row, torch.tensor([.5, 0.]))
    tensor([[0.5000, 1.0000],
            [0.0000, 0.0000]])
    """
    s = scale.to(row.dtype).reshape((row.shape[0], ) + (1, ) * (row.dim() - 1))
    return torch.where(
        s > 0, row * s, torch.zeros((), dtype=row.dtype, device=row.device))


def clip_rows(k: float, *rows: torch.Tensor, ref: torch.Tensor,
              reference: float | None = None) -> tuple:
    """
    Bounded influence on per-board rows, the board first: one norm per
    board over all the rows together, one scale applied to all of them,
    relative to the per-board losses ``ref``. Returns ``(rows, stats)``.

    Parameters:
        k : The clip factor.
        rows : The per-board rows.
        ref : The per-board losses.
        reference : The slow reference, or ``None`` for the batch median.

    Example
    -------
    >>> rows = torch.tensor([[3.], [4.], [40.]])  # doctest: +EXTRA
    >>> (clipped, ), stats = clip_rows(2., rows, ref=torch.ones(3))
    >>> clipped.squeeze(-1), stats["clipped_frac"]
    (tensor([3., 4., 8.]), 0.3333333333333333)
    """
    b = rows[0].shape[0]
    work = torch.float64 if rows[0].dtype == torch.float64 else torch.float32
    squares = sum(r.detach().reshape(b, -1).to(work).pow(2).sum(1)
                  for r in rows)
    norms = torch.sqrt(squares)
    scale = scales(norms, k, ref, reference)
    return (tuple(rescale(r, scale) for r in rows),
            statistics(norms, ref, scale))


class Bound(nn.Module):
    """
    ``fn(net, *args)`` as a module, so that ``functional_call`` swaps the
    parameters of ``net`` underneath it: their names gain the prefix
    ``net.``.

    Parameters:
        net : The module whose parameters ``fn`` reads.
        fn : The function of the module and the arguments.

    Example
    -------
    >>> net = torch.nn.Linear(2, 1)  # doctest: +EXTRA
    >>> bound = Bound(net, lambda net, x: net(x).sum())
    >>> [name for name, _ in bound.named_parameters()]
    ['net.weight', 'net.bias']
    >>> x = torch.ones(1, 2)
    >>> assert torch.equal(bound(x), net(x).sum())
    """
    def __init__(self, net: nn.Module, fn):
        super().__init__()
        self.net, self.fn = net, fn

    def forward(self, *args):
        """ ``fn(net, *args)``. """
        return self.fn(self.net, *args)


def per_board_backward(net: nn.Module, board_loss, batched: tuple, k: float,
                       exclude: tuple = ("halt.", ),
                       reference: float | None = None) -> tuple:
    """
    Per-board parameter gradients, clipped and added to every parameter's
    ``.grad``: what ``backward()`` of the batch *mean* would leave there,
    each board's row scaled by :func:`scales` first.

    ``board_loss(net, *args) -> (loss, aux)`` is ONE board's loss, every
    tensor in ``args`` with a leading batch axis of one; ``batched`` holds
    them ``(B, ...)``. Parameters under the ``exclude`` prefixes are left
    out of the norm and summed unscaled; a parameter the loss never touches
    keeps ``.grad`` ``None``, as autograd would leave it (``torch.func``
    reports zeros there, and AdamW decays a zero gradient but skips a
    missing one), and so does a frozen one, ``requires_grad=False``. The
    norms sum the parameters in registration order.
    Returns ``(aux stacked over the boards, stats)``.

    Parameters:
        net : The module whose parameters get the gradient.
        board_loss : One board's loss and auxiliary outputs.
        batched : The per-board arguments of ``board_loss``.
        k : The clip factor.
        exclude : The name prefixes left out of the norm.
        reference : The slow reference, or ``None`` for the batch median.

    Example
    -------
    >>> net = torch.nn.Linear(2, 1)  # doctest: +EXTRA
    >>> def board_loss(net, x, y):
    ...     return (net(x) - y).pow(2).mean(), (net(x).detach(), )
    >>> x, y = torch.randn(4, 2), torch.randn(4, 1)
    >>> (out, ), stats = per_board_backward(
    ...     net, board_loss, (x, y), k=1e9, exclude=())
    >>> out.shape, stats["clipped_frac"]
    (torch.Size([4, 1, 1]), 0.0)
    >>> grad = torch.autograd.grad((net(x) - y).pow(2).mean(), net.weight)[0]
    >>> assert torch.allclose(net.weight.grad, grad)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    k = positive(k)
    bound = Bound(net, board_loss)
    params = {name: p.detach() for name, p in bound.named_parameters()
              if p.requires_grad}
    buffers = dict(bound.named_buffers())
    counted = {name for name in params
               if not any(name[len("net."):].startswith(e) for e in exclude)}

    def one(params_, *args):
        args = tuple(a.unsqueeze(0) if torch.is_tensor(a) else a for a in args)
        loss, aux = functional_call(bound, (params_, buffers), args)
        loss = loss.reshape(())
        return loss, (loss.detach(), ) + tuple(aux)

    in_dims = (None, ) + (0, ) * len(batched)
    b = batched[0].shape[0]
    rows, aux = vmap(grad(one, has_aux=True), in_dims=in_dims,
                     randomness="same")(params, *batched)
    rows = {name: row.detach() for name, row in rows.items()}
    aux = tuple(a.detach() for a in aux)
    losses, aux = aux[0], aux[1:]
    work = torch.float64 if next(iter(rows.values())).dtype == torch.float64 \
        else torch.float32
    squares = sum(rows[name].reshape(b, -1).to(work).pow(2).sum(1)
                  for name in params if name in counted)
    norms = torch.sqrt(squares)
    losses = losses.to(work)
    scale = scales(norms, k, losses, reference)
    stats = statistics(norms, losses, scale)
    with torch.no_grad():
        for name, p in bound.named_parameters():
            row = rows.get(name)
            if row is None or not bool(row.any()):
                continue
            if name in counted:
                row = rescale(row, scale)
            total = row.sum(0) / b
            p.grad = total if p.grad is None else p.grad + total
    return aux, stats
