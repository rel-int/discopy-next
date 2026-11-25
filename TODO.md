# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.1b and §10.5 of the tutorial "Architectures as Algebra": data augmentation from the same group as
equivariance, slice 4 of [#121](https://github.com/rel-int/discopy-next/issues/121), and the homomorphisms of
*Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332, checked on G-sets.)

- [ ] `sample` and `augment`: a group element drawn by the state monad `kleisli.Seed`, the same element acting on input and label
- [ ] `augmentation`: the augmentation as a network box from a key and the data to a key and the data
- [ ] `symmetrise`: the Reynolds operator, exact over the group or by Monte-Carlo, equal to averaging weights over orbits
- [ ] `homomorphism`, `is_homomorphism`: maps of G-sets as homomorphisms of algebras of `monad.algebra.Writer`
- [ ] tests: purity, uniformity (chi-square), Kleisli composition, Reynolds against orbit means, non-abelian negative controls
- [ ] CHANGELOG bullet
