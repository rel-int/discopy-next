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

## Follow-up experiments: dense maps and whole-call capture

The next round used clean source `e529754c`, the same H100/Torch/CUDA
versions, TRMP revision, seed, weights and benchmark shapes. The later
large-block fallback does not affect these Sudoku shapes. These experiments
do not change the installed backend or global precision settings.

### Whole-call CUDA graphs

Capturing the complete inference or forward/backward call removes much of
the small-batch host overhead. Inference timings include an input device
copy, replay and an output clone; training includes the input copy and
forward/backward replay. Both include synchronization and exclude capture,
compilation and optimizer execution. Both backends are captured, with
networks still in training mode and specialised weight packing inside the
graph. Median milliseconds over ten calls, inference / training:

| Recipe | Batch | Peer kernel, uncaptured | Peer kernel, captured | Specialised, captured |
|---|---:|---:|---:|---:|
| Lattice | 64 | 7.402 / 9.986 | 2.342 / 3.130 | 1.569 / 2.244 |
| ACT | 64 | 8.619 / 16.260 | 3.326 / 6.361 | 2.239 / 4.253 |
| Lattice | 512 | 11.770 / 14.990 | 10.908 / 13.379 | 5.399 / 7.519 |
| ACT | 512 | 16.671 / 29.247 | 15.453 / 26.481 | 8.185 / 15.902 |

The [batch-64 run](https://modal.com/apps/rel-int/main/ap-z6A0jTf7YQRfhvVvVuuZnB)
improves peer execution by 2.56–3.19x, while leaving a fair captured-to-captured
gap of 1.39–1.50x. The
[batch-512 run](https://modal.com/apps/rel-int/main/ap-6YdYPU1vUvA9U6b8jPuxuD)
improves peer execution by only 1.08–1.12x; its remaining gap is 1.67–2.02x.
All batch-512 backends used the same recorded GPU UUID. Wall-clock latency
varies across runs, so these speedups compare measurements within each run.

All sixteen graph cases passed exact comparisons with their own uncaptured
backend, including alternating inputs and nonuniform changes to
`cell.encode.0.weight`. Retained inference outputs stayed unchanged, and
gradient storage stayed stable without accumulation across replay. This
checks capture equivalence; the independent numerical gates above still
apply. The prototype assumes fixed shapes and storage. External
`zero_grad(set_to_none=True)` and optimizer integration are not validated.

A [batch-64 profile](https://modal.com/apps/rel-int/main/ap-KUHd7JsRiKVDSBmAt60zQv)
measured peer inference at 9.196 ms wall / 2.296 ms summed device time for
lattice, and 10.668 / 3.283 ms for ACT. Captured inference approaches those
device sums. This supports addressing host launches at small batches;
large batches still need less device work.

### Tensor-core affine maps

The prototype evaluates `Y = X W.T + b` with Triton's `tf32x3` products and
FP32 accumulation. Its first version also computes `dX` and deterministic
split-K `dW` with tensor cores. A separate profile confirmed that AOTAutograd
actually executes those custom gradient kernels. Twenty-four standalone
checks passed, including FP64 comparisons, sliced/transposed inputs,
static/dynamic compilation, long gradient reductions and fallbacks.

The [first full-network run](https://modal.com/apps/rel-int/main/ap-8A9zim63RbLiWvs48z0KVw)
nevertheless regressed batch-512 inference/training: lattice increased from
11.763 / 14.711 to 12.708 / 15.595 ms, and ACT from 16.562 / 28.750 to
19.006 / 31.229 ms. ACT readout failed the unchanged gate at `3.32e-4`;
gradients passed. A
[FP64 probe](https://modal.com/apps/rel-int/main/ap-MKQCyto4FZ89fEGdN1zdoP)
on its worst board (299) found `2.50e-4` maximum error for the dense
prototype versus `8.23e-5` for eager FP32. This version is not adopted.

A [tile sweep](https://modal.com/apps/rel-int/main/ap-tkhB8hhKTfehj3qeeOgSNn)
checked 216 combinations: eight Triton tiles plus PyTorch, four maps,
forward/input-gradient/weight-gradient, and two batches. Each device timing
is the median of three rotated 100 ms CUDA-graph samples. At batch 512,
the best state projection improved from 34.49 to 27.12 microseconds
(`64/64/64`, four warps), and the gate projection from 30.99 to 25.21
(`128/64/32`, four warps). Most custom input gradients were slower than
PyTorch. These isolated gains justify testing selective forward maps with
PyTorch backward; they do not establish an end-to-end improvement.

The [selective follow-up](https://modal.com/apps/rel-int/main/ap-SrGFE14IakKhwJja9JAHKZ)
uses tensor cores only for the state and gate projections, with PyTorch
backward. The state tile is `64/64/64`; the gate tile is `64/32/32` at
batch 64 and `128/64/32` at batch 512, all with four warps. These choices
come from the timing sweep, not the readout errors. All four network
readouts and parameter gradients pass the unchanged gates. Timings:

| Recipe | Batch | Peer kernel | Selective dense maps |
|---|---:|---:|---:|
| Lattice | 64 | 5.122 / 7.455 | 5.607 / 7.724 |
| ACT | 64 | 6.318 / 11.965 | 6.727 / 13.402 |
| Lattice | 512 | 11.637 / 14.534 | 10.400 / 13.388 |
| ACT | 512 | 16.421 / 28.558 | 15.307 / 27.189 |

The selective candidate is 1.05–1.12x faster at batch 512 and regresses
batch-64 latency by 4–12%. Its maximum readout/gradient errors are
`2.37e-4` / `3.81e-6`. On board 299, its maximum error against FP64 is
`1.55e-4`, versus `1.46e-5` for the peer-only kernel. This is one diagnostic
board and one full-network seed, not a general accuracy bound. Together
with the shape-specific tile policy and mixed latency results, this keeps
the dense-map implementation experimental. Whole-round fusion remains the
main opportunity for reducing large-batch device work.

The [combined comparison](https://modal.com/apps/rel-int/main/ap-s4eSqWD4YVLiJtvzLecOYF)
captures all three backends on the same recorded H100. Median milliseconds,
with the same input-copy/output-ownership scopes above:

| Recipe | Batch | Captured peer kernel | Captured selective maps | Captured specialised |
|---|---:|---:|---:|---:|
| Lattice | 64 | 2.341 / 3.142 | 2.165 / 2.976 | 1.571 / 2.227 |
| ACT | 64 | 3.336 / 6.376 | 3.029 / 5.994 | 2.226 / 4.272 |
| Lattice | 512 | 10.953 / 13.443 | 9.537 / 12.016 | 5.426 / 7.543 |
| ACT | 512 | 15.557 / 26.637 | 14.176 / 25.001 | 8.204 / 15.983 |

Selective maps add 1.06–1.15x over captured peer-only execution. The gap
to captured specialised execution is now 1.34–1.40x at batch 64 and
1.56–1.76x at batch 512. All 24 capture checks pass exactly against their
own uncaptured backend, and source hashes match. Two preceding attempts
retain failed checks for stale autograd stream references. The final helper
warms and captures on one explicit stream and measures uncaptured training
on an identical control model, so the captured model never runs backward
on the default stream first. It does not suppress PyTorch's stream check.

The production change from this round is the large-block fallback described
above, including traceable symbolic block sizes. It passed 36 H100 checks
and the local suite of 2,185 tests (196 skips), with repository lint/type
gates passing. Prototype sources, notebook and raw results are preserved in
the workspace's experiment artifact bundle; neither graph capture nor the
dense prototype is enabled by the library's `kernel="triton"` option.
