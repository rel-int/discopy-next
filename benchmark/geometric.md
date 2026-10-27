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

CUDA float32 inputs with blocks of at most 1024 points use Triton; other
devices, dtypes, larger blocks and layouts requiring 64-bit indices use
PyTorch. The block limit keeps each forward tile within Triton's
1,048,576-element tensor limit.
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

Before timing, outputs and parameter gradients must be finite and gradient
presence must match. Full-network errors and tolerance checks are recorded
for every backend, including failed comparisons. Results are saved before
the command fails if either geometric implementation misses a tolerance.
The specialised baseline's status is reported independently.
The readout gate is `atol=3e-4, rtol=1e-4`, the parameter-gradient gate is
`atol=1e-5, rtol=1e-3`, and isolated pairs use `atol=1e-6, rtol=1e-6`.
The relative pairs allowance covers a measured Triton gradient discrepancy
of `1.19e-6` at magnitude `2.88`; the readout allowance covers the compiled
PyTorch ACT baseline's `2.42e-4` discrepancy from eager at batch 512.
The JSON records maximum absolute and relative L2 errors, tolerances,
hardware/software versions, both repositories' revisions and dirty flags,
and SHA-256 hashes of the kernel, geometric implementation, wiring and runner.
Source identity is captured before launch, and a TRMP revision other than
`a768520` is rejected. Timing alone is not a parity check.

Modal requires both its control-plane host `api.modal.com` and its regional
input-plane host (observed here as `input-plane.us-east.modal.com`) to be
reachable. The local kernel tests run without Modal:

```sh
UV_CACHE_DIR=/workspace/.uv-cache uv run --group grammar pytest \
    test/neural/kernels.py test/neural/geometric.py
```

## Measured network performance

The [complete H100 run](https://modal.com/apps/rel-int/main/ap-AubmqVh3RdkqQF4epJeUqs)
on 2026-10-27 used clean commit `0821d13f` and TRMP `a768520`.
Median milliseconds per call, inference / training, with ten repetitions:

| Recipe | Batch | Compiled geometric | Fused Triton | Specialised TRMP |
|---|---:|---:|---:|---:|
| Lattice | 64 | 6.67 / 9.48 | 8.08 / 11.45 | 2.47 / 3.75 |
| ACT | 64 | 10.95 / 17.48 | 10.37 / 14.35 | 3.05 / 5.40 |
| Lattice | 512 | 32.20 / 38.78 | 11.74 / 14.58 | 6.38 / 8.73 |
| ACT | 512 | 51.17 / 79.77 | 16.51 / 28.90 | 9.40 / 17.33 |

At batch 512, fused geometric is 2.66–3.10 times faster than compiled
geometric, and still 1.67–1.84 times slower than the specialised kernel.
Batch 64 is mixed: lattice regresses by about 21%, while ACT improves.
The kernel remains opt-in; these measurements do not establish parity or
an improvement for every workload. Folded-layout peer reduction alone at
batch 512 takes 0.106 / 0.365 ms versus PyTorch's 0.491 / 1.415 ms.

Both geometric implementations pass the declared accuracy gates. Across
these four cases, fused maximum absolute readout/gradient errors are
`9.69e-5` / `2.86e-6`, and maximum relative L2 errors are `1.69e-6` /
`2.18e-7`. The specialised ACT baseline fails three readout elements at
batch 512 (`7.07e-4` maximum absolute, `5.20e-6` relative L2 error);
its gradients pass. The JSON reports this failure independently, without
relaxing the library gate.

A [float64 diagnostic](https://modal.com/apps/rel-int/main/ap-o8LjZpieQWHZC8wUlwxSQd)
on the worst board (299) found maximum errors of `8.23e-5` for eager fp32,
`1.60e-4` for compiled geometric, `1.46e-5` for fused, and `6.25e-4` for
specialised. This checks one sensitive board, not the entire input domain.

## H100 validation and tuning

CPU checks cover an independent dense relation, signed overlapping blocks,
gradients, ReLU boundaries, noncontiguous tensors, empty tensors and full
network/checkpoint compatibility. CUDA tests additionally cover full-graph
compilation, custom-operator registration contracts and nonfinite inputs.

The final configuration uses `TILE=1024, num_warps=4` for forward and
`TILE=256, num_warps=2` for backward. The H100 correctness run at clean commit
`0821d13f` passed all 35 tests, including explicit dynamic-layout compilation
and the CPU block-axes regression. Both Modal hosts were reachable.

The [first sweep](https://modal.com/apps/rel-int/main/ap-R8WK7c72LKe3GIcML7pTQ7)
compared tiles 64, 128 and 256 with 2, 4 and 8 warps, at batches 64 and 512,
for both layouts and both forward and backward. All 72 combinations passed
the finite-value and PyTorch accuracy checks; `256/2` won that grid.
Both sweeps used an H100 80GB, Torch 2.13.0/CUDA 13.0 and Triton 3.7.1.
Each result is the median of three 100 ms CUDA-graph measurements, rotating
configuration order between measurements.
`triton.testing.do_bench_cudagraph` excludes Python dispatch and allocation;
these device timings cannot establish whole-network speedup.

First-stage folded-layout kernel times in microseconds, comparing the original
`128/4` launch with `256/2`:

| Batch | Forward 128/4 | Forward 256/2 | Backward 128/4 | Backward 256/2 |
|---|---:|---:|---:|---:|
| 64 | 42.17 | 6.52 | 86.34 | 11.91 |
| 512 | 318.98 | 40.71 | 660.85 | 84.34 |

The [extended sweep](https://modal.com/apps/rel-int/main/ap-ceXbmtg1lsScjqqwMuIbZg)
validated another 64 combinations with the same protocol. Forward `1024/4`
won both layouts at both batches, reducing folded-layout time from 6.53 to
5.18 microseconds at batch 64 and from 40.82 to 30.87 at batch 512.
Backward retains `256/2`: `512/4` improved only about 3–5%, while the
folded-layout winner `1024/8` regressed contiguous inputs.

A separate [profile](https://modal.com/apps/rel-int/main/ap-9o01nf4MgzoOHDoJRCkR3M)
of intermediate `256/2` commit `878da558`, before the final forward tuning,
measured 10.87 ms of device work for lattice inference at batch 512:
5.91 ms in matrix products, 2.36 ms in fused GRU/normalization, and 2.44 ms
in peer reductions. Matrix products and GRU/normalization account for 76%
of that device time; TRMP's specialised whole-segment implementation took
5.51 ms of device work. The profile identifies work beyond peer reduction
that remains to close the gap; the final timings above measure the whole
call separately.
