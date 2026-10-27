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

CUDA float32 inputs use Triton; other devices and dtypes use PyTorch.
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
uv tool install 'modal[api-proxy-support]'
modal run benchmark/geometric_modal.py --mode tests
modal run benchmark/geometric_modal.py --batches 64,512 --repeat 10 \
    --out geometric-results.json
```

`TRMP_SOURCE` can point to an existing checkout at that revision. The runner
mounts the current DisCoPy source and pins Torch 2.13/CUDA 13.0 on an H100.
It compares the compiled PyTorch geometry, compiled fused geometry, and
TRMP's specialised CUDA segment, with the same weights, inputs and loss.
Lattice and ACT recipes use the shapes in TRMP's `bench.py`. It also isolates
the peer reduction. Timings are median wall times after warm-up, with CUDA
synchronization and compilation excluded; training includes backward.
The JSON includes readout and parameter-gradient errors against the eager
reference. Timing alone is not a parity check.

Modal requires both its control-plane host `api.modal.com` and its regional
input-plane host (observed here as `input-plane.us-east.modal.com`) to be
reachable. The local kernel tests run without Modal:

```sh
uv run pytest test/neural/kernels.py test/neural/geometric.py
```

## Validation status

CPU checks cover an independent dense relation, signed overlapping blocks,
gradients, ReLU boundaries, noncontiguous tensors, empty tensors and full
network/checkpoint compatibility. CUDA tests additionally cover full-graph
compilation, custom-operator registration contracts and nonfinite inputs.

Offline H100 (`sm90`) compilation passed for the forward and backward
kernels on four geometries. Triton's CPU interpreter also passed 32
comparisons against the dense reference: those geometries, widths 1, 7, 35
and 76, forward and backward, with strided inputs, absolute tolerance
`1e-6` and no relative tolerance. These checks validate compilation and
arithmetic; they do not validate execution or performance on a GPU.

H100 validation and performance measurements are pending: this environment
successfully built the benchmark image, but its network policy currently
blocks the regional Modal input plane. No speedup or parity with TRMP's
specialised kernel is claimed until those tests and measurements run.
