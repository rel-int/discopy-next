# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.1 of the tutorial "Architectures as Algebra": equivariant layers realised without the dense tied
matrix, here along the construction of the group from symmetric groups by products and wreath products; the
equivariant networks of *Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.)

- [ ] `structure`: the factors of an action of a product, wreath or power of symmetric groups, sizes checked first
- [ ] `Kronecker`: the layer as a sum over tuples of orbits of poolings and broadcasts along the axes
- [ ] `realise`: Pooling, then Kronecker, then Dense
- [ ] tests: equal to Dense in outputs and gradients, same numbering, no orbit of pairs at scale
- [ ] CHANGELOG bullet
