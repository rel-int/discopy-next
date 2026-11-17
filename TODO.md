> Take a look at our ongoing work on monoidal egraphs (discopy-next) and the work that Alessandro had started (wiki + upstream discopy): are we missing anything? it seems like Alessandro had an intuition about a tabular data structure for string diagrams that could alleviate some of the limitations we are encountering.
>
> write the issue brief for the join matcher
>
> (slice 2 of rel-int/discopy-next#512; default taken 2026-11-16 by the 48 h rule)

- [x] Builder.rewrite and Builder.step: a match rewritten by add and union, a round rewritten then rebuilt
- [x] EGraph.saturate on the builder, a freeze after each round that changes something
- [x] EGraph.reduced, one spider per class and one box per e-node, and law stepping
- [x] the closed form of + on n leaves as a test: 63 classes and 602 e-nodes at n = 6 in under 2 s
- [x] CHANGELOG bullets under Added and Changed
- [x] the gates of CONTRIBUTING.md's Run the unit tests
