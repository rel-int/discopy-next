# -*- coding: utf-8 -*-

"""Fused peer reductions against a dense relation and its derivatives."""

from itertools import product
from math import prod

from pytest import importorskip, mark, raises, skip


def geometry(kind):
    from discopy.neural.geometric import Peers
    from discopy.neural.wiring import Block
    shape, terms = {
        "complete": ((3, 5), ((1, ()), (-1, (0, 1)))),
        "columns": ((2, 3, 4), ((1, (0, 2)), (-1, (0, 1, 2)))),
        "self": ((3, 5), ((1, (0, 1)),)),
        "sudoku": ((3, 3, 3, 3), (
            (1, (0, 1)), (1, (2, 3)), (1, (0, 2)),
            (-1, (0, 1, 2)), (-1, (0, 2, 3)), (-1, (0, 1, 2, 3)))),
    }[kind]
    blocks = tuple((c, Block(shape, keep)) for c, keep in terms)
    return Peers(blocks, sum(c * b.size for c, b in blocks))


def dense(torch, peers, s, x):
    coordinates = tuple(product(*(range(n) for n in peers.terms[0][1].shape)))
    relation = torch.tensor([
        [sum(c for c, block in peers.terms
             if all(v[a] == w[a] for a in block.keep))
         for w in coordinates] for v in coordinates],
        dtype=s.dtype, device=s.device)
    messages = torch.relu(s.unsqueeze(2) + x.unsqueeze(1))
    return (messages * relation[None, :, :, None]).sum(2) / peers.degree


def tensors(torch, peers, width, device, dtype, exact=False, strided=False):
    n = prod(peers.terms[0][1].shape)
    generator = torch.Generator(device=device).manual_seed(718)
    shape = (3, width, n) if strided else (3, n, width)
    values = [torch.randn(shape, generator=generator, device=device,
                          dtype=dtype) for _ in range(2)]
    if exact:
        values = [v.round() for v in values]
    if strided:
        values = [v.transpose(1, 2) for v in values]
    return tuple(v.requires_grad_() for v in values)


def check(torch, peers, s, x, compiled=False, cotangent=None, dynamic=None):
    reference = dense(torch, peers, s, x)
    call = lambda s, x: peers(s, x, kernel="triton")
    if compiled:
        call = torch.compile(call, fullgraph=True, dynamic=dynamic)
    actual = call(s, x)
    tolerance = 1e-12 if s.dtype == torch.float64 else 1e-6
    torch.testing.assert_close(actual, reference, rtol=0, atol=tolerance)
    if cotangent is None:
        cotangent = torch.linspace(-2, 3, actual.numel(), device=s.device,
                                   dtype=s.dtype).reshape_as(actual)
    expected_grad = torch.autograd.grad(reference, (s, x), cotangent)
    actual_grad = torch.autograd.grad(actual, (s, x), cotangent)
    for found, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(found, expected, rtol=0, atol=tolerance)


@mark.parametrize("kind", ["complete", "columns", "self", "sudoku"])
def test_cpu_fallback(kind):
    torch = importorskip("torch")
    peers = geometry(kind)
    for dtype in (torch.float32, torch.float64):
        for exact in (False, True):
            s, x = tensors(torch, peers, 7, "cpu", dtype, exact, strided=True)
            check(torch, peers, s, x)
    s, x = tensors(torch, geometry("self"), 1, "cpu", torch.float64)
    assert torch.autograd.gradcheck(
        lambda s, x: geometry("self")(s, x, kernel="triton"), (s, x))


def cuda():
    torch = importorskip("torch")
    if not torch.cuda.is_available():
        skip("CUDA is required for the Triton peer kernel")
    importorskip("triton")
    return torch


@mark.parametrize("kind", ["complete", "columns", "self", "sudoku"])
@mark.parametrize("width", [1, 35, 128])
def test_cuda_pairs(kind, width):
    torch, peers = cuda(), geometry(kind)
    for exact in (False, True):
        s, x = tensors(torch, peers, width, "cuda", torch.float32,
                       exact, strided=True)
        check(torch, peers, s, x)


def test_cuda_compile():
    torch, peers = cuda(), geometry("sudoku")
    s, x = tensors(torch, peers, 35, "cuda", torch.float32, strided=True)
    check(torch, peers, s, x, compiled=True)


def test_cuda_folded_layout():
    """Symbolic sliced-map strides and a noncontiguous cotangent."""
    torch, peers = cuda(), geometry("sudoku")
    generator = torch.Generator(device="cuda").manual_seed(718)
    s, x = [torch.randn(3, 81, width, device="cuda", generator=generator)
            for width in (76 + 3 * 39 + 15, 76)]
    s, x = s[..., :76].requires_grad_(), x.requires_grad_()
    cotangent = torch.randn(3, 76, 81, device="cuda", generator=generator)
    cotangent = cotangent.transpose(1, 2)
    assert s.stride() != x.stride() and not cotangent.is_contiguous()
    check(torch, peers, s, x, compiled=True, cotangent=cotangent, dynamic=True)


def test_cuda_double_fallback():
    torch, peers = cuda(), geometry("columns")
    s, x = tensors(torch, peers, 7, "cuda", torch.float64)
    check(torch, peers, s, x)


@mark.parametrize("device", ["cpu", "cuda"])
def test_network(device):
    torch = cuda() if device == "cuda" else importorskip("torch")
    from discopy.group import Symmetric
    from discopy.neural.geometric import Geometric
    from discopy.neural.recursion import PeerNet

    class Complete(Geometric, PeerNet):
        group = Symmetric[5]

    torch.manual_seed(42)
    peers = tuple(tuple(w for w in range(5) if w != v) for v in range(5))
    dtype = torch.float32 if device == "cuda" else torch.float64
    ref = Complete((4, 6, 8, 5), peers, 4, 2, 3, 1).to(device, dtype)
    net = Complete((4, 6, 8, 5), peers, 4, 2, 3, 1,
                   kernel="triton").to(device, dtype)
    net.kernel = "geometric"
    assert net.peer_kernel == "triton"
    net.load_state_dict(ref.state_dict(), strict=True)
    x = torch.rand(2, 5, 4, device=device, dtype=dtype)
    s, t = ref.initial(x), net.initial(x)
    cycle = torch.compile(net.cycle, fullgraph=True) \
        if device == "cuda" else net.cycle
    tolerance = 1e-5 if device == "cuda" else 1e-12
    for _ in range(2):
        s, t = ref.cycle(s), cycle(t)
        for expected, actual in zip(s, t):
            torch.testing.assert_close(actual, expected,
                                       rtol=0, atol=tolerance)
    target = torch.randn_like(s.state)
    (s.state * target).sum().backward()
    (t.state * target).sum().backward()
    for p, q in zip(ref.parameters(), net.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad,
                                       rtol=0, atol=tolerance)


def test_validation():
    torch = importorskip("torch")
    peers = geometry("self")
    s, x = tensors(torch, peers, 1, "cpu", torch.float32)
    with raises(ValueError, match="kernel"):
        peers(s, x, kernel="unknown")
    for invalid in (peers._replace(terms=()), peers._replace(degree=0),
                    peers._replace(degree=-1)):
        with raises(ValueError, match="terms and positive degree"):
            invalid(s, x)
    for a, b in ((s[0], x[0]), (s, x[:1]), (s, x.double()),
                 (s[:, :2], x[:, :2]), (s, x.to("meta"))):
        with raises(ValueError, match="matching"):
            peers(a, b)


def test_fake_registration():
    torch = importorskip("torch")
    from torch._subclasses.fake_tensor import FakeTensorMode
    from discopy.neural.kernels import encode, peer_pairs
    peers = geometry("columns")
    with FakeTensorMode():
        s = torch.empty(2, 24, 7)
        result = peer_pairs(s, s, encode(peers.terms), peers.degree)
        assert result.shape == s.shape and result.dtype == s.dtype
        assert result.device == s.device


@mark.parametrize("backward", [False, True])
def test_offline_compile(backward, tmp_path, monkeypatch):
    """Compile real CUDA code without requiring a CUDA device."""
    torch, triton = importorskip("torch"), importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from discopy.neural.kernels import decode, encode, reduction

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    peers = geometry("sudoku")
    s, x = tensors(torch, peers, 35, "cpu", torch.float32, strided=True)
    terms = decode(encode(peers.terms))
    source = ASTSource(
        reduction(), signature={
            **dict.fromkeys(("S", "X", "G", "Y", "DX"), "*fp32"),
            "LENGTH": "i32"},
        constexprs={"N": s.shape[1], "H": s.shape[2],
                    "TERMS": terms, "DEGREE": peers.degree,
                    "BACKWARD": backward, "TILE": 256,
                    **{f"{prefix}{i}": stride
                       for prefix, t in (("SS", s), ("XS", x))
                       for i, stride in enumerate(t.stride())}})
    compiled = triton.compile(
        source, target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 2, "enable_fp_fusion": False})
    assert compiled.asm["cubin"]


@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("shape", [(0, 15, 7), (2, 15, 0)])
def test_empty_inputs(device, shape):
    torch = cuda() if device == "cuda" else importorskip("torch")
    peers = geometry("complete")
    for kernel in ("torch", "triton"):
        s, x = [torch.empty(shape, device=device, requires_grad=True)
                for _ in range(2)]
        result = peers(s, x, kernel=kernel)
        assert result.shape == s.shape and not result.numel()
        grads = torch.autograd.grad(result.sum(), (s, x))
        assert all(g.shape == s.shape and not g.numel() for g in grads)


def test_cuda_opcheck():
    torch, peers = cuda(), geometry("columns")
    from discopy.neural.kernels import encode, peer_pairs
    s, x = tensors(torch, peers, 7, "cuda", torch.float32)
    torch.library.opcheck(
        peer_pairs, (s, x, encode(peers.terms), peers.degree))


def test_cuda_nonfinite():
    torch, peers = cuda(), geometry("columns")
    s = torch.tensor([float("nan"), float("inf"), -float("inf"), 0., 1., -1.],
                     device="cuda").repeat(4).reshape(1, 24, 1)
    s, x = s.requires_grad_(), s.detach().flip(1).requires_grad_()
    expected, actual = peers(s, x), peers(s, x, kernel="triton")
    torch.testing.assert_close(actual, expected, equal_nan=True)
    grads = [torch.autograd.grad(y, (s, x), torch.ones_like(y))
             for y in (expected, actual)]
    for found, reference in zip(grads[1], grads[0]):
        torch.testing.assert_close(found, reference, equal_nan=True)
