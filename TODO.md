Clone [https://github.com/rel-int/TRMP](https://github.com/rel-int/TRMP) then implement a kernel for discopy-next that tries to match the performance of Tommaso's specialised kernel [https://github.com/rel-int/discopy-next/issues/377](https://github.com/rel-int/discopy-next/issues/377)

Mathematical design: for the signed block relation A(v,w), compute
M(v) = sum_w A(v,w) relu(s(v) + x(w)) / degree. Fuse block reductions,
and differentiate the same sum with respect to each input using the
symmetric block relation, without atomic gradient accumulation.

- [x] Implement an opt-in fused Peers kernel and portable fallback.
- [x] Check values, gradients, layouts and compiled network integration.
- [x] Benchmark against the PyTorch geometry and TRMP specialised kernel;
      document measured performance and remaining limitations.
- [WIP] @codex-h100-2026-10-27 10:44 Run repository checks and prepare the change for review.

Previous round: 2181 passed with --skip-extra; focused CPU checks, pflake8,
pylint, ruff and ty pass. Eight offline H100 compilations and 32 Triton
interpreter comparisons pass (forward/backward atol=1e-6, rtol=0).
Completed H100 run: https://modal.com/apps/rel-int/main/ap-AubmqVh3RdkqQF4epJeUqs
Clean source 0821d13f; TRMP a768520; both Modal endpoints reachable.
35 tests passed on H100. Both geometric backends pass accuracy gates.
Forward uses TILE=1024/warps=4, backward TILE=256/warps=2, after two
measured sweeps. Dynamic layout/geometry tracing and large-index fallback
are validated; Block.free preserves symbolic axis membership.

Batch 512, inference / training ms:
- Lattice: geometric 32.20 / 38.78, fused 11.74 / 14.58,
  specialised 6.38 / 8.73.
- ACT: geometric 51.17 / 79.77, fused 16.51 / 28.90,
  specialised 9.40 / 17.33.
Fused is 2.66–3.10x faster than geometric but 1.67–1.84x slower than
specialised. Batch 64 is mixed: lattice regresses about 21%; ACT improves.
Fused max readout error 9.69e-5, gradient 2.86e-6. Specialised ACT has
three readout outliers (max 7.07e-4); its independent status is recorded.
The worst-board fp64 probe finds fused max error 1.46e-5 vs specialised
6.25e-4. Full results are documented in benchmark/geometric.md and saved
at /workspace/discopy-artifacts/geometric-results.json with source hashes.

Local suite: 2,185 passed, 195 skipped; lint/type gates passed (pylint 9.18).
The latest target branch is merged. CI lint, unit tests and coverage pass;
documentation/notebook jobs and final-head CI remain. Keep the PR draft
until CI and review gates finish.

Continuation request (verbatim):

Continue implementing and benchmarking the discopy-next kernel for:
https://github.com/rel-int/discopy-next/issues/377

Existing draft PR:
https://github.com/rel-int/discopy-next/pull/393
Branch: codex/geometric-pairs-kernel

The goal is to approach Tommaso’s specialised TRMP kernel’s performance. You may use the configured Modal credentials for H100 tests and benchmarks.

Workspace:
- /workspace/discopy-next — implementation and repository instructions
- /workspace/TRMP — original reference repository
- /workspace/TRMP-bench — benchmark revision a768520

Implemented:
- Opt-in Geometric(..., kernel="triton"), stored as peer_kernel to avoid conflicting with TRMP’s kernel policy.
- Generic fused signed-block peer reduction and backward, strided inputs, CPU/non-float32 fallback.
- Tests and benchmark/geometric_modal.py.

Validation already passed:
- 2,181 local tests, 188 skips.
- Latest focused tests: 27 passed, 19 CUDA skips.
- Eight offline H100 compilations.
- 32 Triton interpreter comparisons within absolute tolerance 1e-6.
- Repository lint/type-check gates.

No actual GPU execution or performance measurement has succeeded yet. The previous session authenticated to Modal and built its image, but the running network policy blocked input-plane.us-east.modal.com with HTTP 403. I updated the environment configuration; verify this fresh session allows both api.modal.com and input-plane.us-east.modal.com.

Read AGENTS.md and the cloud-environment skill, inspect the existing branch, then run:

/workspace/bin/modal run benchmark/geometric_modal.py --mode all --batches 64,512 --repeat 10 --out /workspace/discopy-artifacts/geometric-results.json

Fix GPU failures, measure accuracy and performance against the existing compiled geometric implementation and TRMP’s specialised kernel using identical weights, and tune where justified. Update the PR, benchmark documentation and TODO.md with actual results. Keep performance claims grounded in measured GPU results; the current documentation explicitly records validation as pending.

Use UV_CACHE_DIR=/workspace/.uv-cache for local uv commands. Never print credentials.

Experiment request (verbatim):

let's go with another round of experiments!

Dense-map design: Y = X W^T + b, dX = dY W, dW = dY^T X and
db = sum_rows dY. Evaluate the products with three TF32 tensor-core
products and fp32 accumulation, retaining the fp32 fallback and checking
full recurrent values and gradients. Compare against the peer-only kernel
with identical weights; profile batch 64 before choosing further fusion.

- [x] Experiment with tensor-core dense maps and gradients, profile small
      batches, and record measured decisions and results.

Follow-up measurements use clean e529754c and TRMP a768520. Whole-call
graphs improve peer-only batch 64 execution 2.56–3.19x, but both captured
backends still differ by 1.39–1.50x. Batch 512 gains only 1.08–1.12x;
its fair captured gap is 1.67–2.02x. Changing inputs and a matrix weight,
retained outputs and stable/nonaccumulating gradients are checked.

All-map tf32x3 regresses batch 512 and fails ACT readout at 3.32e-4.
The 216-case tile sweep and 24 standalone checks pass. Selective state/gate
forward maps with PyTorch backward pass current full-network gates and
are 1.05–1.12x faster at batch 512, while regressing batch 64. FP64 probes retain
their larger rounding error. Prototypes remain experimental; no global
precision change or default-backend change is made.

The final combined graph run passes all 24 checks on the same H100:
https://modal.com/apps/rel-int/main/ap-s4eSqWD4YVLiJtvzLecOYF
Selective maps add 1.06–1.15x over captured peer-only execution; remaining
captured-specialised gaps are 1.34–1.40x at batch 64 and 1.56–1.76x at
batch 512. Warmup/capture share a stream; uncaptured training is timed on
an identical control model to avoid stale autograd stream references.

Production review found a Triton tensor-size limit for blocks >1024.
The fallback and symbolic Block.size fix are committed and pushed.
Final checks: 36 passed on H100; 2,185 local passed / 196 skipped;
repository lint/type gates pass. Previous head 5ec4e850 CI passed;
final-head CI remains to be verified. Review found no other actionable
issue and no GitHub review thread awaits an agent. Keep the draft until
the remaining review/CI gate above completes.

Raw results and experiment sources are in /workspace/discopy-artifacts;
the notebook and archive preserve reproduction commands and source hashes.
The repository-directed wiki checkout was rejected by automatic approval
review as outside this task's scope, so prototype artifacts remain local.
