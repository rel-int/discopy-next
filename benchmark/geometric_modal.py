"""
Reproduce the H100 comparison with TRMP's specialised sudoku segment.

Check out rel-int/TRMP at a768520 in ../TRMP-bench (or set TRMP_SOURCE),
install ``modal[api-proxy-support]``, then run from the library checkout::

    modal run benchmark/geometric_modal.py --mode smoke
    modal run benchmark/geometric_modal.py --mode tests
    modal run benchmark/geometric_modal.py --batches 64,512 --repeat 10

Use ``--mode all`` to run correctness checks before measuring on the same GPU.

The image pins the torch/CUDA versions used in issue #377. Sources are
mounted from this checkout; the specialised CUDA build is cached in a
Modal volume. Results include readout and parameter-gradient errors;
timings are wall-clock medians after two warm-ups, including synchronising
CUDA, with compilation excluded. No reduced-precision global flags are set.
As in TRMP's benchmark, nets remain in training mode during inference;
the specialised kernel therefore includes weight packing on every call.
Accuracy gates apply to both geometric implementations; the specialised
baseline's errors and pass/fail status are reported independently.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
TRMP = Path(os.environ.get("TRMP_SOURCE", ROOT.parent / "TRMP-bench"))
WIDTHS = (15, 39, 76, 20)
# The fused pairs gradient differs by 1.19e-6 at magnitude 2.88
# on H100, so its gate includes a relative rounding allowance.
# The compiled PyTorch ACT readout differs from eager by 2.42e-4 at b512.
TOLERANCES = {"readout": {"rtol": 1e-4, "atol": 3e-4},
              "gradient": {"rtol": 1e-3, "atol": 1e-5},
              "pairs": {"rtol": 1e-6, "atol": 1e-6}}
app = modal.App("discopy-geometric-pairs")
cache = modal.Volume.from_name(
    "discopy-geometric-kernels", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04",
                              add_python="3.12")
    .pip_install("torch==2.13.0", "numpy==2.5.2", "ninja", "networkx",
                 "matplotlib", "pillow", "huggingface_hub==1.28.0",
                 "pytest", "hypothesis")
    .env({"PYTHONPATH": "/repo:/trmp", "GNN_KERNEL_DIR": "/cache/cuda",
          "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
          "TRITON_CACHE_DIR": "/cache/triton", "MAX_JOBS": "4"})
    .add_local_dir(ROOT / "discopy", "/repo/discopy",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(ROOT / "test/neural", "/repo/test/neural",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(TRMP, "/trmp",
                   ignore=[".git", "**/__pycache__/**", ".venv"]))


def accuracy(actual, expected, kind, strict=True):
    """Require finite values; retain errors and failed comparisons."""
    import torch
    assert torch.isfinite(expected).all(), f"nonfinite {kind} reference"
    assert torch.isfinite(actual).all(), f"nonfinite {kind} result"
    passed = torch.allclose(actual, expected, **TOLERANCES[kind])
    if strict:
        torch.testing.assert_close(actual, expected, **TOLERANCES[kind])
    delta = actual.double() - expected.double()
    scale = torch.linalg.vector_norm(expected.double()).item()
    return {"passed": passed, "max_abs": delta.abs().max().item(),
            "relative_l2": torch.linalg.vector_norm(delta).item() / scale
            if scale else None}


def compare(case, batch, repeat):
    """The same model, weights, inputs and loss on each implementation."""
    import torch
    from bench import SHAPES, act_calls, clock, lattice_calls
    from sudoku import eval_act, eval_lattice

    device = torch.device("cuda")
    torch.manual_seed(0)
    x = (torch.rand(batch, 81, 9, device=device) > 0.5).float()
    halt = case.endswith("act")
    build = eval_act.build if halt else eval_lattice.build
    calls = act_calls if halt else lattice_calls

    def make(path):
        return build(WIDTHS, *SHAPES[case], kernel=path).to(device)

    reference = make("off")
    reference.kernel = "off"
    expected = calls(reference, x)[0]()
    calls(reference, x)[1]()
    gradients = {k: p.grad.detach().clone() for k, p in
                 reference.named_parameters() if p.grad is not None}
    results = {}
    for name in ("geometric", "triton", "specialised"):
        print(f"Checking {case}, batch {batch}, {name}", flush=True)
        torch._dynamo.reset()
        net = make("auto" if name == "specialised" else "geometric")
        net.load_state_dict(reference.state_dict())
        if name == "specialised":
            net.kernel = "fp32"
        else:
            net.peer_kernel = "triton" if name == "triton" else "torch"
        if name != "specialised":
            net.compile(unroll=False, fullgraph=True)
        inference, training = calls(net, x)
        errors = {"readout": accuracy(inference(), expected, "readout", False)}
        training()
        actual = {k: p.grad for k, p in net.named_parameters()
                  if p.grad is not None}
        assert actual.keys() == gradients.keys(), "gradient presence differs"
        errors["gradient"] = accuracy(
            torch.cat([actual[k].reshape(-1) for k in gradients]),
            torch.cat([g.reshape(-1) for g in gradients.values()]),
            "gradient", False)
        row = {"inference_ms": 1000 * clock(inference, repeat, device),
               "training_ms": 1000 * clock(training, repeat, device),
               **{f"{kind}_{metric}": value for kind, metrics in errors.items()
                  for metric, value in metrics.items()}}
        results[name] = row
        print(json.dumps({"case": case, "batch": batch, "backend": name,
                          **row}), flush=True)
        del net
    return results


def pairs_benchmark(batch, repeat, folded=False):
    """Peer aggregation, contiguous or with the folded maps' exact strides."""
    import torch
    from bench import SHAPES, clock
    from sudoku import eval_lattice

    device = torch.device("cuda")
    torch.manual_seed(1)
    net = eval_lattice.build(WIDTHS, *SHAPES["peer"], kernel="geometric")
    wiring = net.geometry
    s, x = [torch.randn(batch, 81, 76, device=device, requires_grad=True)
            for _ in range(2)]
    if folded:
        maps = net.maps()
        widths = (sum(maps.state_sizes), sum(maps.beliefs_sizes))
        s, x = [torch.empty(batch, 81, width, device=device)[..., :76]
                .copy_(t.detach()).requires_grad_()
                for t, width in zip((s, x), widths)]
    dy = torch.randn_like(s)
    expected = wiring(s, x)
    gradients = torch.autograd.grad(expected, (s, x), dy)
    results = {}
    for kernel in ("torch", "triton"):
        apply = torch.compile(lambda a, b: wiring(a, b, kernel=kernel),
                              fullgraph=True)

        def forward():
            with torch.no_grad():
                return apply(s, x)

        def backward():
            torch.autograd.grad(apply(s, x), (s, x), dy)

        output_error = accuracy(forward(), expected, "pairs", False)
        actual = torch.autograd.grad(apply(s, x), (s, x), dy)
        gradient_error = accuracy(torch.stack(actual), torch.stack(gradients),
                                  "pairs", False)
        results[kernel] = {
            "inference_ms": 1000 * clock(forward, repeat, device),
            "training_ms": 1000 * clock(backward, repeat, device),
            "output_error": output_error, "gradient_error": gradient_error}
    results["strides"] = {"s": s.stride(), "x": x.stride()}
    print(json.dumps({"case": "pairs_folded" if folded else "pairs",
                      "batch": batch, **results}), flush=True)
    return results


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=3600,
              volumes={"/cache": cache})
def run(mode: str, batches: str, repeat: int, source: dict):
    """Run correctness checks or both sudoku recipes on one GPU."""
    import torch
    sys.path[:0] = ["/repo", "/trmp"]
    info = {"gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "batch_sizes": batches,
            "repeat": repeat, "tolerances": TOLERANCES,
            "inference_training_mode": True, **source}
    print(json.dumps(info), flush=True)
    if mode in ("tests", "all"):
        tests = [str(p) for p in Path("/repo/test/neural").glob("*kernel*.py")]
        tests.append("/repo/test/neural/geometric.py")
        tests.append("/repo/test/neural/wiring.py::test_dynamic_block_axes")
        subprocess.run([sys.executable, "-m", "pytest", "-c", "/dev/null",
                        "--import-mode=importlib", "-q", *tests], check=True)
    if mode in ("bench", "all"):
        info["results"] = {
            str(batch): {"pairs": pairs_benchmark(batch, repeat),
                         "pairs_folded": pairs_benchmark(batch, repeat, True),
                         **{case: compare(case, batch, repeat)
                            for case in ("peer", "peer-act")}}
            for batch in map(int, batches.split(","))}
        info["accuracy_passed"] = {
            name: all(batch[case][name]["readout_passed"]
                      and batch[case][name]["gradient_passed"]
                      for batch in info["results"].values()
                      for case in ("peer", "peer-act"))
            for name in ("geometric", "triton", "specialised")}
        info["accuracy_passed"]["pairs"] = all(
            batch[case][kernel][error]["passed"]
            for batch in info["results"].values()
            for case in ("pairs", "pairs_folded")
            for kernel in ("torch", "triton")
            for error in ("output_error", "gradient_error"))
    elif mode not in ("smoke", "tests"):
        raise ValueError("mode must be smoke, tests, bench or all")
    cache.commit()
    return info


@app.local_entrypoint()
def main(mode: str = "bench", batches: str = "64,512", repeat: int = 10,
         out: str = "geometric-results.json"):
    """Write the remote result with the exact local revision recorded."""
    source = {}
    for name, path in (("discopy", ROOT), ("trmp", TRMP)):
        source[f"{name}_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        source[f"{name}_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=path, text=True))
    if not source["trmp_revision"].startswith("a768520"):
        raise ValueError("TRMP_SOURCE must be checked out at a768520")
    source["source_sha256"] = {
        str(path.relative_to(ROOT)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "discopy/neural/kernels.py",
                     ROOT / "discopy/neural/geometric.py",
                     ROOT / "discopy/neural/wiring.py", Path(__file__))}
    result = run.remote(mode, batches, repeat, source)
    Path(out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not all(result.get("accuracy_passed", {}).get(name, True)
               for name in ("geometric", "triton", "pairs")):
        raise ValueError(
            f"geometric accuracy gate failed; results saved to {out}")
