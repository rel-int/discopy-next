> Fix the categorical-deep-learning notebook, which fails on main since #737 (build runs on 9bd4d1bf, fc26c034, 8b744b50: "Expected feedback.Ty, got monoidal.Ty instead"). Found while watching main's CI after #772 merged.

- [x] The constant parameter stream and its Stream codomain use the plain markov category, as stream.Ty.sequence does
