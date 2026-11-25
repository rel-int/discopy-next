# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.2b and §10.3 of the tutorial "Architectures as Algebra": codatatypes and their unfolds, the final
coalgebras of *Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332, App. J.2–J.5,
stacked on the branch of `neural.datatype`.)

- [x] `Machine`: a cell `I @ S.d -> O @ S` with its feedback, the Mealy, Moore and unfolding cells of the paper's Figure 1
- [x] `Machine.unroll`: the first steps through monoidal streams, the initial state one input copied where it is read
- [x] `Coalgebra`: one image per box of the cell and a learned initial state, tied by one copy of the parameter
- [x] tests against independent Python loops, a torch RNN with the gradient of its shared weight, and `Network.trace`
- [x] CHANGELOG bullet
