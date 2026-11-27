`lens.Prism`, the prisms `(match, build)` over a comarkov category, the
dual of `lens.Markov`: a symmetric lens whose residual is what `match`
fails on, merged with what `build` builds, defined as a Markov lens over
the opposite category `op.Markov`, e.g. the constructors of a sum type
over `python.additive.Function` ([next#42](https://github.com/rel-int/discopy-next/issues/42)).
