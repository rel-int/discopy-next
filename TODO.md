> Take a look at our ongoing work on monoidal egraphs (discopy-next) and the work that Alessandro had started (wiki + upstream discopy): are we missing anything? it seems like Alessandro had an intuition about a tabular data structure for string diagrams that could alleviate some of the limitations we are encountering.
>
> write the issue brief for the join matcher

(slice 1 of rel-int/discopy-next#512; default taken 2026-11-16 by the 48 h rule)

- [x] Builder.matches: a hash join over keys, box_wires and heads, with an index by head filled lazily when a match asks
- [x] law matching_agreement as an Axiom, enrolled in the property matrix
- [x] unit tests in test/egraph.py, including Copy(x) >> g
- [x] measure the join against EGraph.matches on the largest trace
- [x] CHANGELOG bullet under Added
- [x] the gates of CONTRIBUTING.md's Run the unit tests
