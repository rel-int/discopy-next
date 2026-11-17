= What is left <sec:left>

This draft says where it stops: each block marked as not written yet
names something that the library, its proofs or this text lack. They fall
in four groups.

- *Modules to write.* `discopy.peirce`, with its `Formula` and its tested
  axioms (@sec:peirce). The loaders that reproduce two published parsers
  from their weights (@sec:lambek). A reducer for interaction nets, and
  rules for Boolean circuits short enough to list (@sec:lafont). The
  recursive networks as a `Network` with a trace, where they are PyTorch
  modules with a loop (@sec:neural-recursion). The translation of a
  lambda term to a `Network`, and the transformer that it gives, with
  its embedding and a training run to report (@sec:neural-example).
- *Laws and proofs.* The freeness of `Hypergraph`, `CMap` and `Term` in
  Lean, as it is proved of `Diagram`, and colours in that model and in
  the terms that the property matrix draws (@sec:lean, @sec:regions).
  `Drawing` in the property matrix with an equality up to isotopy, the
  isotopy between two drawings computed and animated, and reading a
  diagram back from a picture (@sec:drawing). A strategy for the
  intertwiners of a Hopf algebra (@sec:representations).
- *Drawing.* Regions and frames in the Typst and TikZ backends, without
  which the coloured regions of @sec:regions and the e-graphs of
  @sec:rewriting have no picture here.
- *Text.* The scaling plots and the timings of the benchmarks
  (@sec:benchmarks, @sec:circuits). The references that the library cites
  without their details, among them those of optyx, of the Hopf algebras
  and of the recursive networks. The title, the authors and the release
  to cite (@sec:build).
