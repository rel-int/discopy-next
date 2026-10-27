Clone [https://github.com/rel-int/TRMP](https://github.com/rel-int/TRMP) then implement a kernel for discopy-next that tries to match the performance of Tommaso's specialised kernel [https://github.com/rel-int/discopy-next/issues/377](https://github.com/rel-int/discopy-next/issues/377)

Mathematical design: for the signed block relation A(v,w), compute
M(v) = sum_w A(v,w) relu(s(v) + x(w)) / degree. Fuse block reductions,
and differentiate the same sum with respect to each input using the
symmetric block relation, without atomic gradient accumulation.

- [x] Implement an opt-in fused Peers kernel and portable fallback.
- [WIP] @codex-h100-2026-10-27 10:44 Check values, gradients, layouts and compiled network integration.
- [WIP] @codex-h100-2026-10-27 10:44 Benchmark against the PyTorch geometry and TRMP specialised kernel;
      document measured performance and remaining limitations.
- [WIP] @codex-h100-2026-10-27 10:44 Run repository checks and prepare the change for review.

Local checks: 2181 passed with --skip-extra; focused CPU checks, pflake8,
pylint, ruff and ty pass. Eight offline H100 compilations and 32 Triton
interpreter comparisons pass (forward/backward atol=1e-6, rtol=0).
GPU tests and timing remain blocked by the network policy denying
input-plane.us-east.modal.com (HTTP 403). The user has been asked to add
this required Modal hostname. Keep this change in draft until GPU
validation, performance measurements and CI have completed.
