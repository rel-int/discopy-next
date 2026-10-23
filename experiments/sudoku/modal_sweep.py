"""Modal app running the sudoku experiments on GPUs.

The image carries the dependencies of the ``optyx`` extra of this
discopy branch plus the JAX GPU stack; the discopy checkout and this
experiment directory are mounted so the smoke test can contract an
``interaction.CMap`` unrolling on the GPU through ``eval`` under
``discopy.tensor.backend("jax")``, and the training runs use the batched
Cotengra/JAX contraction of ``contraction.py``.

Run, from ``experiments/sudoku``::

    modal run modal_app.py::smoke_test
    modal run modal_app.py --configs configs.json --out results.json
"""

import json
import pathlib
import sys

import modal

app = modal.App("optyx-sudoku")

try:
    REPO = pathlib.Path(__file__).resolve().parents[2]
except IndexError:                       # inside the container
    REPO = pathlib.Path("/root/discopy")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]==0.4.38", "optax", "cotengra", "quimb", "numpy",
        "opt_einsum", "networkx", "matplotlib", "pillow", "sympy",
        "pylatexenc>=2.10", "perceval-quandela>=1.3", "graphix>=0.4",
        "pytket>=2.6.0", "pytket-pyzx>=0.36", "pytket-qiskit[aer]>=0.68.3",
        "pyzx==0.10.7",
    )
    .env({"PYTHONPATH": "/root/discopy:/root/discopy/experiments/sudoku"})
    .add_local_dir(
        REPO, remote_path="/root/discopy",
        ignore=[".venv", ".git", "docs", "**/__pycache__"])
)

GPU = "A100-40GB"


@app.function(image=image, gpu=GPU, timeout=1200)
def smoke_test():
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    devices = [str(device) for device in jax.devices()]

    from discopy.tensor import backend
    from discopy.optyx.channel import qubit
    from discopy.optyx.interaction import Box, CMap
    from discopy.optyx.qubits import Z, X, Scalar, Ket

    cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
    cmap = CMap([Box("f", qubit, qubit, cnot)], [((0, 0), (0, 1))])
    unrolled = (Ket(0) @ Ket(0)) >> cmap.unroll(1)
    network = unrolled.double().to_tensor().to_map()
    with backend("jax"):
        gpu_result = network.eval()
    cpu_result = network.eval()
    assert np.allclose(
        np.asarray(gpu_result.array), np.asarray(cpu_result.array))
    return {
        "devices": devices,
        "gpu_matches_cpu": True,
        "result": np.asarray(gpu_result.array).ravel().tolist(),
    }


@app.function(image=image, gpu=GPU, timeout=6 * 3600)
def run_config(config: dict) -> dict:
    sys.path.insert(0, "/root/discopy/experiments/sudoku")
    import experiment as ex
    import contraction as co

    train_cases, test_cases = ex.make_dataset()
    result, _ = co.train(config, train_cases, test_cases)
    return result


@app.local_entrypoint()
def main(configs: str = "configs.json", out: str = "results.json"):
    with open(configs) as handle:
        config_list = json.load(handle)
    results = []
    for result in run_config.map(config_list, order_outputs=False):
        results.append(result)
        with open(out, "w") as handle:
            json.dump(results, handle, indent=2)
        final = result["final"] or {}
        print(result["config"], result["n_parameters"],
              final.get("cell_accuracy"), final.get("solve_rate"),
              flush=True)
