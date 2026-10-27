# -*- coding: utf-8 -*-

"""
Fused CUDA reductions for :class:`discopy.neural.geometric.Peers`.

For the symmetric signed block relation ``A``, the forward map is
``sum_w A[v, w] relu(s[v] + x[w]) / degree``. Its two vector-Jacobian
products reduce over the same blocks, with the arguments exchanged for
the sender gradient. Each output has one writer: no scatter or atomics.
The activation mask is recomputed, so no edge activations are stored.

The Triton operators expose their launches to ``torch.compile`` so compiled
networks avoid Python dispatch per round. Triton kernels are compiled lazily.
Only float32 and first derivatives are supported; use the geometric
PyTorch implementation for other dtypes and higher derivatives.
"""

from functools import cache
from math import prod

import torch


@cache
def reduction():
    """ The lazily compiled forward and backward reduction. """
    import triton  # pylint: disable=import-outside-toplevel
    import triton.language as tl  # pylint: disable=import-outside-toplevel

    @triton.jit
    def run(S, X, G, Y, DX, N: tl.constexpr, H: tl.constexpr,
            LENGTH, TERMS: tl.constexpr, DEGREE: tl.constexpr,
            BACKWARD: tl.constexpr, TILE: tl.constexpr,
            SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr,
            XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr):
        # pylint: disable=invalid-name,too-many-locals
        offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
        valid = offset < LENGTH
        v = offset // H % N
        batch = offset // (N * H)
        channel = offset % H
        s_address = batch * SS0 + v * SS1 + channel * SS2
        x_address = batch * XS0 + v * XS1 + channel * XS2
        s = tl.load(S + s_address, valid, other=0)
        total = tl.full((TILE,), 0, tl.float32)
        if BACKWARD:
            x = tl.load(X + x_address, valid, other=0)
            grad = tl.load(G + offset, valid, other=0)
            sender = tl.full((TILE,), 0, tl.float32)
        for term in tl.static_range(len(TERMS)):
            coefficient = TERMS[term][0]
            size = TERMS[term][1]
            member = tl.arange(
                0, triton.next_power_of_2(TERMS[term][1]))[:, None]
            w = v[None, :] + tl.full(member.shape, 0, tl.int32)
            divisor = 1
            for axis in tl.static_range((len(TERMS[term]) - 2) // 2):
                stride = TERMS[term][2 + 2 * axis]
                extent = TERMS[term][3 + 2 * axis]
                w = w + (member // divisor % extent
                         - v[None, :] // stride % extent) * stride
                divisor *= extent
            mask = valid[None, :] & (member < size)
            address = (batch[None, :] * XS0 + w * XS1
                       + channel[None, :] * XS2)
            other = tl.load(X + address, mask, other=0)
            value = s[None, :] + other
            if BACKWARD:
                active = tl.where(mask & ~(value <= 0), grad[None, :], 0)
                address = (batch[None, :] * SS0 + w * SS1
                           + channel[None, :] * SS2)
                other_s = tl.load(S + address, mask, other=0)
                address = (batch[None, :] * N + w) * H + channel[None, :]
                other_grad = tl.load(G + address, mask, other=0)
                incoming = tl.where(
                    mask & ~(other_s + x[None, :] <= 0), other_grad, 0)
                sender += coefficient * tl.sum(incoming, 0)
            else:
                active = tl.where(mask & ~(value <= 0), value, 0)
            total += coefficient * tl.sum(active, 0)
        tl.store(Y + offset, total / DEGREE, valid)
        if BACKWARD:
            tl.store(DX + offset, sender / DEGREE, valid)

    return run


def launch(s, x, terms, degree, grad=None):
    """ Reduce strided inputs using encoded block axes. """
    descriptors = decode(terms)
    out = torch.empty_like(s, memory_format=torch.contiguous_format)
    dx = torch.empty_like(out) if grad is not None else out
    if s.numel():
        with torch.cuda.device(s.device):
            torch.library.wrap_triton(reduction())[
                ((s.numel() + 255) // 256,)](
                s, x, grad if grad is not None else s, out, dx,
                s.shape[1], s.shape[2], s.numel(), descriptors, degree,
                grad is not None, 256, *s.stride(), *x.stride(),
                num_warps=2, enable_fp_fusion=False)
    return out, dx


@torch.library.triton_op("discopy::peer_pairs", mutates_args=())
def peer_pairs(s: torch.Tensor, x: torch.Tensor,
               terms: list[int], degree: int) -> torch.Tensor:
    """ The fused mean over a signed block relation. """
    return launch(s, x, terms, degree)[0]


@peer_pairs.register_fake
def fake_pairs(s, x, terms, degree):
    """ The output has the receiver's shape and dtype. """
    # pylint: disable=unused-argument  # custom operator signature
    return torch.empty_like(s, memory_format=torch.contiguous_format)


@torch.library.triton_op("discopy::peer_pairs_backward", mutates_args=())
def peer_pairs_backward(s: torch.Tensor, x: torch.Tensor, grad: torch.Tensor,
                        terms: list[int], degree: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """ Both vector-Jacobian products, without atomic accumulation. """
    return launch(s, x, terms, degree, grad.contiguous())


@peer_pairs_backward.register_fake
def fake_backward(s, x, grad, terms, degree):
    """ The gradients have the inputs' shapes and dtypes. """
    # pylint: disable=unused-argument  # custom operator signature
    return tuple(torch.empty_like(t, memory_format=torch.contiguous_format)
                 for t in (s, x))


def setup_context(ctx, inputs, output):
    """ Save inputs, not an activation for every edge. """
    # pylint: disable=unused-argument  # custom operator signature
    s, x, ctx.terms, ctx.degree = inputs
    ctx.save_for_backward(s, x)


def backward(ctx, grad):
    """ Differentiate the same symmetric relation for each argument. """
    ds, dx = peer_pairs_backward(
        *ctx.saved_tensors, grad, ctx.terms, ctx.degree)
    return ds, dx, None, None


peer_pairs.register_autograd(backward, setup_context=setup_context)


def encode(terms):
    """ Encode each block by coefficient, axis count and strides/sizes. """
    return [value for coefficient, block in terms
            for value in (coefficient, len(block.free),
                          *(value for axis in reversed(block.free)
                            for value in (prod(block.shape[axis + 1:]),
                                          block.shape[axis])))]


def decode(terms):
    """ Specialize the geometry before nesting its compile-time constants. """
    terms = [int(value) for value in terms]
    descriptors = []
    while terms:
        coefficient, count = terms[:2]
        axes = terms[2:2 + 2 * count]
        descriptors.append((coefficient, prod(axes[1::2]), *axes))
        terms = terms[2 + len(axes):]
    return tuple(descriptors)
