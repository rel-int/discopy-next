# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.1 of the tutorial "Architectures as Algebra": equivariant layers realised without the dense tied
matrix, here for cyclic groups, i.e. convolutions; the equivariant networks of *Categorical Deep Learning is an
Algebraic Theory of All Architectures*, arXiv:2402.15332.)

- [x] `structure`: cyclic factors, alone or in products with symmetric ones
- [x] `Kronecker`: correlation along cyclic axes, direct below `fft_from` points, by FFT from there
- [x] tests: both paths equal Dense in outputs and gradients, double and single precision, numbering of weights
- [x] CHANGELOG bullet
