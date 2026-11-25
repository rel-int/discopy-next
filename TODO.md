# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.1 of the tutorial "Architectures as Algebra": equivariant layers as boxes, slice 3 of
[#121](https://github.com/rel-int/discopy-next/issues/121), the homomorphisms of algebras of the monad G × − of
*Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.)

- [ ] `group.Action.trivial`: the action of a group fixing every point
- [ ] `neural.equivariant.tying`: the orbit of each entry of a matrix from the orbits of `cod.tensor(dom)`
- [ ] `Dense`: the reference layer with one weight per orbit, channels and an equivariant bias
- [ ] `layer`: the parametric box of `para.Markov[Network]`, `equivariance`: the check on generators
- [ ] tests against an independent orbit labelling, the Bell numbers, MLP / CNN / Deep Sets weight counts, negative controls
- [ ] CHANGELOG bullet
