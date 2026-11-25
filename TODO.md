# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.1 of the tutorial "Architectures as Algebra": equivariant layers realised without the dense tied
matrix, here for symmetric groups on tuples of points, by set partitions; the equivariant networks of *Categorical
Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.)

- [x] `Partitions`: the layers of S_n between its natural action and its tensor square, one weight per set partition
- [x] `realise`: Pooling, Kronecker, then Partitions, then Dense
- [x] tests: equal to Dense in outputs and gradients, numbering, the partitions counted, refusals
- [x] CHANGELOG bullet
