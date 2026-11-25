# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.5 of the tutorial "Architectures as Algebra": the 2-cells of Para, reparametrisations r: P' -> P
acting contravariantly on parameters, under which the laws of parametric maps hold, as in *Categorical Deep
Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332, where Para is a 2-category.)

- [ ] `para.Cell`: a reparametrisation 2-cell between parallel maps, with identity, vertical and horizontal composition
- [ ] `para.UpToPermutation`: maps equal up to an invertible 2-cell, decided by an isomorphism of hypergraphs that fixes the boundary
- [ ] the laws declared failing up to reparametrisation now hold modulo `UpToPermutation`, in every para category
- [ ] tests: negatives (output swaps, box data, port order, parameter vs boundary ports, adjoints), bounded searches
- [ ] CHANGELOG bullet
