# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.7 of the tutorial "Architectures as Algebra", second part: the diagram of a recursive message-passing
network compiled by a functor that stacks the boxes sharing a module, so that the drawn network runs as fast as the
hand-written table; *Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.)

- [ ] `stack`: the boxes sharing a module and a type go to one box on a node axis, the routing to one gather
- [ ] buckets in the order the diagram draws them, a bucket of one node without a node axis
- [ ] tests: on regular graphs the stacked round IS `FactorNet.round` (`torch.equal`), mixed degrees to 1e-12, gradients through the unrolled trace, one module call per bucket per round
- [ ] CHANGELOG bullet
