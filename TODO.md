# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.1 of the tutorial "Architectures as Algebra": equivariant layers realised without the dense tied
matrix, here the interface for such realisations and the first one, pooling to or from a single point; the
equivariant networks of *Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.)

- [x] `neural.equivariant.Realisation`: the interface of a layer's realisation, `Dense` one of them, `Legs` holding one
- [x] `neural.factorised.Pooling`: the layers from or to a single point, without the dense matrix
- [x] `realise` and `layer(..., fast=True)`: Pooling where it applies, Dense otherwise; the default unchanged
- [x] tests: equal to Dense in outputs and gradients, same numbering of weights, the fallback never raises
- [x] CHANGELOG bullet
