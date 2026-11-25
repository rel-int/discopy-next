# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.7 of the tutorial "Architectures as Algebra": a recursive message-passing network drawn as a diagram,
the factor graph read as a closed map in the Int construction, one trace whose unrolled rounds are those of
`Recursion.round`, as in *Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.)

- [ ] `neural.diagram`: `interpret` a closed map as one round, `of` a PeerNet or FactorNet, `closed`, `legs`
- [ ] `wiring.ports` and `wiring.where` factored out of `from_map`, behaviour unchanged
- [ ] tests: the round is `Recursion.round`, the trace unrolls to a cycle, same parameters, equal gradients through the unrolled trace, float64
- [ ] refusals: a node among its own peers, an asymmetric peers table, a factor named like the nodes
- [ ] CHANGELOG bullet
