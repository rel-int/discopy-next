> Build rel-int/discopy-next#42, prisms: lenses over a comarkov category, as USER (toumix) ruled live 11-17: the prism is `lens.Prism`; #511's `lens.Comarkov` keeps its name.

- [ ] `lens.Prism`: `(match, build)` over a comarkov category, defined as the Markov lens over its opposite `op.Markov`, a symmetric lens with `dom.negative` as residual
- [ ] doctest: the `Left` prism of a sum type over `python.additive.Function`
- [ ] unit tests in `test/lens.py`, the lens cells of the property matrix for `lens.Prism`
- [ ] CHANGELOG `Added` bullet, next-checks green
