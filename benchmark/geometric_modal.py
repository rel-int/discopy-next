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
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
TRMP = Path(os.environ.get("TRMP_SOURCE", ROOT.parent / "TRMP-bench"))
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
        return build((15, 39, 76, 20), *SHAPES[case], kernel=path).to(device)

    reference = make("off")
    reference.kernel = "off"
    expected = calls(reference, x)[0]()
    calls(reference, x)[1]()
    gradients = {k: p.grad.detach().clone() for k, p in
                 reference.named_parameters() if p.grad is not None}
    results = {}
    for name in ("geometric", "triton", "specialised"):
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
        error = (inference() - expected).abs().max().item()
        training()
        gradient_error = max((p.grad - gradients[k]).abs().max().item()
                             for k, p in net.named_parameters()
                             if k in gradients)
        row = {"inference_ms": 1000 * clock(inference, repeat, device),
               "training_ms": 1000 * clock(training, repeat, device),
               "readout_max_abs": error, "gradient_max_abs": gradient_error}
        results[name] = row
        print(json.dumps({"case": case, "batch": batch, "backend": name,
                          **row}), flush=True)
        del net
    return results


def pairs_benchmark(batch, repeat):
    """Isolate peer aggregation and its backward from maps and recursion."""
    import torch
    from bench import clock
    from sudoku import geometric, data
    from discopy.neural.geometric import Peers

    device = torch.device("cuda")
    torch.manual_seed(1)
    wiring = Peers.of(data.PEERS, geometric.GROUP)
    s, x = [torch.randn(batch, 81, 76, device=device, requires_grad=True)
            for _ in range(2)]
    dy = torch.randn_like(s)
    results = {}
    for kernel in ("torch", "triton"):
        apply = torch.compile(lambda a, b: wiring(a, b, kernel=kernel),
                              fullgraph=True)

        def forward():
            with torch.no_grad():
                return apply(s, x)

        def backward():
            torch.autograd.grad(apply(s, x), (s, x), dy)

        results[kernel] = {
            "inference_ms": 1000 * clock(forward, repeat, device),
            "training_ms": 1000 * clock(backward, repeat, device)}
    print(json.dumps({"case": "pairs", "batch": batch, **results}), flush=True)
    return results


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=3600,
              volumes={"/cache": cache})
def run(mode: str, batches: str, repeat: int):
    """Run correctness checks or both sudoku recipes on one GPU."""
    import torch
    sys.path[:0] = ["/repo", "/trmp"]
    info = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "batch_sizes": batches,
            "repeat": repeat, "trmp_revision": "a768520"}
    print(json.dumps(info), flush=True)
    if mode in ("tests", "all"):
        tests = [str(p) for p in Path("/repo/test/neural").glob("*kernel*.py")]
        tests.append("/repo/test/neural/geometric.py")
        subprocess.run([sys.executable, "-m", "pytest", "-c", "/dev/null",
                        "--import-mode=importlib", "-q", *tests], check=True)
    if mode in ("bench", "all"):
        info["results"] = {
            str(batch): {"pairs": pairs_benchmark(batch, repeat),
                         **{case: compare(case, batch, repeat)
                            for case in ("peer", "peer-act")}}
            for batch in map(int, batches.split(","))}
    elif mode not in ("smoke", "tests"):
        raise ValueError("mode must be smoke, tests, bench or all")
    cache.commit()
    return info


@app.local_entrypoint()
def main(mode: str = "bench", batches: str = "64,512", repeat: int = 10,
         out: str = "geometric-results.json"):
    """Write the remote result with the exact local revision recorded."""
    result = run.remote(mode, batches, repeat)
    result["discopy_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["discopy_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True))
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
