# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.3 of the tutorial "Architectures as Algebra": `Network.trace` read as a stream of rounds, the
feedback semantics of the unfolds of *Categorical Deep Learning is an Algebraic Theory of All Architectures*,
arXiv:2402.15332, as an opt-in beside today's constant-input unrolling.)

- [x] `Unroll(rounds, sequence=True)`: one input and one output per round, the state threaded from zeros
- [x] `Network.unroll`, `to_function` and `to_lens` pass the option through; the default unchanged
- [x] nested traces merged by the vanishing law into one memory of several wires; other shapes refused clearly
- [x] tests: the old semantics as the new one on a constant input, hand-written loops, backpropagation through time
- [x] CHANGELOG bullet
