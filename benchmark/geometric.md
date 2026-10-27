# Fused geometric peer reductions

Issue [#377](https://github.com/rel-int/discopy-next/issues/377) attributes
about 70% of the geometric Sudoku network's H100 time to separate signed
block reductions. `Geometric(..., kernel="triton")` fuses those reductions
and their backward pass. The default `kernel="torch"` remains the reference.

For blocks with coefficients `c`, let `A[v,w]` be the sum of the
coefficients of blocks containing both points. The operation is

```
y[v,h] = sum_w A[v,w] relu(s[v,h] + x[w,h]) / degree.
```

The backward pass recomputes the ReLU masks. Since each block is symmetric,
the sender derivative is another reduction over the same blocks with the
roles exchanged. Each output has one writer, so gradients need no atomics
and no tensor with an activation per edge. Block axes and sizes are compile
time constants; the kernel is independent of Sudoku's wiring. Input strides
are passed directly, including slices of the folded matrix products.

`torch.library.triton_op` and `wrap_triton` expose the launches to
`torch.compile`, avoiding Python operator dispatch on every compiled round.
The six input strides cross this boundary as separate scalar arguments;
tuple-valued strides failed when tracing symbolic layouts.

CUDA float32 inputs use Triton; other devices, dtypes and layouts requiring
64-bit indices use PyTorch.
The fused operator supports first derivatives. Use `kernel="torch"` for
higher derivatives. The implementation changes neither matrix multiplication
precision nor the carried state's dtype. It does not fuse whole rounds or
segments; matrix products, GRUs, normalization and refresh remain PyTorch
operations.

## Reproduce the comparison

Use the same benchmark revision as issue #377, which contains the geometric
adapters. The main branch of TRMP has since reorganized its kernel modules.

```sh
git clone https://github.com/rel-int/TRMP ../TRMP-bench
git -C ../TRMP-bench checkout a768520
UV_CACHE_DIR=/workspace/.uv-cache uv tool install 'modal[api-proxy-support]'
modal run benchmark/geometric_modal.py --mode all --batches 64,512 --repeat 10 \
    --out geometric-results.json
```

`TRMP_SOURCE` can point to an existing checkout at that revision. The runner
mounts the current DisCoPy source and pins Torch 2.13/CUDA 13.0 on an H100.
It compares the compiled PyTorch geometry, compiled fused geometry, and
TRMP's specialised CUDA segment, with the same weights, inputs and loss.
Lattice and ACT recipes use the shapes in TRMP's `bench.py`. It also isolates
the peer reduction with contiguous inputs and with the folded maps' strides:
the receiver's 76 channels occupy a slice of width 208, while the sender's
width is 76. Both layouts share the same sampled values across backends.
Timings are median wall times after two warm-ups, including CUDA
synchronization but excluding compilation; training includes backward.
As in issue #377, networks remain in training mode during inference, so
TRMP repacks weights on each call. These are not `.eval()` inference timings
with cached weights.

Before timing, outputs and parameter gradients must be finite, gradient
presence must match, and values must pass `torch.testing.assert_close`.
The readout gate is `atol=3e-4, rtol=1e-4`, the parameter-gradient gate is
`atol=1e-5, rtol=1e-3`, and isolated pairs use `atol=1e-6, rtol=1e-6`.
The relative pairs allowance covers a measured Triton gradient discrepancy
of `1.19e-6` at magnitude `2.88`; the readout allowance covers the compiled
PyTorch ACT baseline's `2.42e-4` discrepancy from eager at batch 512.
The JSON records maximum absolute and relative L2 errors, tolerances,
hardware/software versions, both repositories' revisions and dirty flags,
and SHA-256 hashes of the kernel, geometric implementation and runner.
Source identity is captured before launch, and a TRMP revision other than
`a768520` is rejected. Timing alone is not a parity check.

Modal requires both its control-plane host `api.modal.com` and its regional
input-plane host (observed here as `input-plane.us-east.modal.com`) to be
reachable. The local kernel tests run without Modal:

```sh
UV_CACHE_DIR=/workspace/.uv-cache uv run --group grammar pytest \
    test/neural/kernels.py test/neural/geometric.py
```

## H100 validation and tuning

CPU checks cover an independent dense relation, signed overlapping blocks,
gradients, ReLU boundaries, noncontiguous tensors, empty tensors and full
network/checkpoint compatibility. CUDA tests additionally cover full-graph
compilation, custom-operator registration contracts and nonfinite inputs.

The latest H100 correctness run passed all 34 tests with scalar stride
arguments and the selected launch configuration. Both Modal hosts were
reachable. Earlier offline compilation and 32 Triton interpreter comparisons
also passed; GPU execution is now validated directly.

The fixed configuration is `TILE=256, num_warps=2`, selected after measuring
tiles 64, 128 and 256 with 2, 4 and 8 warps, at batches 64 and 512, for
both layouts and both forward and backward. All 72 combinations passed the
finite-value and PyTorch accuracy checks; `256/2` was fastest in every case.
The [completed Modal sweep](https://modal.com/apps/rel-int/main/ap-R8WK7c72LKe3GIcML7pTQ7)
used the final scalar-stride signature on an H100 80GB, Torch 2.13.0/CUDA 13.0
and Triton 3.7.1. Each result is the median of three 100 ms CUDA-graph
measurements, rotating configuration order between measurements.
`triton.testing.do_bench_cudagraph` excludes Python dispatch and allocation;
these device timings cannot establish whole-network speedup.

Folded-layout kernel times in microseconds, comparing the original `128/4`
launch with the selected `256/2`:

| Batch | Forward 128/4 | Forward 256/2 | Backward 128/4 | Backward 256/2 |
|---|---:|---:|---:|---:|
| 64 | 42.17 | 6.52 | 86.34 | 11.91 |
| 512 | 318.98 | 40.71 | 660.85 | 84.34 |

These results establish a launch-configuration improvement for the measured
reduction. They do not establish parity with TRMP's specialised whole-segment
kernel; full-network measurements are still running.
