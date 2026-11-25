# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.6 of the tutorial "Architectures as Algebra": parametric maps over an actegory, weights in one
monoidal category acting on data in another, as in *Categorical Deep Learning is an Algebraic Theory of All
Architectures*, arXiv:2402.15332, where Para is defined for an actegory and Thm G.10 forces weight tying to be the
diagonal when the weights are cartesian.)

- [x] `actegory.Action`: a strict action of a monoidal category on another, its laws as equations, `right` and `deterministic`
- [x] `actegory.Para`: identity, composition, reparametrisation over an action
- [x] tests: the self-action is `para`, the laws on deterministic weights acting on stochastic data, non-actions caught
- [x] weight tying is reparametrisation by copy: natural for deterministic maps, not for a sampled weight
- [x] docs and CHANGELOG bullet
