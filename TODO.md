Clone [https://github.com/rel-int/TRMP](https://github.com/rel-int/TRMP) then implement a kernel for discopy-next that tries to match the performance of Tommaso's specialised kernel [https://github.com/rel-int/discopy-next/issues/377](https://github.com/rel-int/discopy-next/issues/377)

Mathematical design: for the signed block relation A(v,w), compute
M(v) = sum_w A(v,w) relu(s(v) + x(w)) / degree. Fuse block reductions,
and differentiate the same sum with respect to each input using the
symmetric block relation, without atomic gradient accumulation.

- [ ] Implement an opt-in fused Peers kernel and portable fallback.
- [ ] Check values, gradients, layouts and compiled network integration.
- [ ] Benchmark against the PyTorch geometry and TRMP specialised kernel;
      document measured performance and remaining limitations.
- [ ] Run repository checks and prepare the change for review.
