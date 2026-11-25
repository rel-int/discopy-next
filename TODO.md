# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

("This" is §10.2 of the tutorial "Architectures as Algebra": datatypes and their folds, the missing ingredient
for the initial algebras of *Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.)

- [x] `discopy.neural.datatype`: `Datatype` given by its constructors, its terms as diagrams built by composition and tensor
- [x] `Algebra`: one image per constructor, the fold as a functor, tied by one copy of the parameter per constructor (App. J)
- [x] tests against independent recursions: lists and trees into Python functions, free Markov diagrams and torch networks
- [x] gradient of the shared weight equals backpropagation through the recursion
- [x] CHANGELOG bullet
