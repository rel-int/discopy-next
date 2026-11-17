> Build the first slice of rel-int/discopy-next#10, as USER (toumix) ruled: on decision 5 of #56 (11-09, 'yes to both… `Sum` its own module (#10)'), the enrichment in commutative monoids becomes one module `discopy.sum` parametrised by the underlying category the way `discopy.para` is, and `bubble` stays in the hierarchy; and live 11-17, the per-level `Sum` classes become `deprecated_alias`es of the generic one (not subclasses), sliced by level.

- [x] discopy/sum.py: the generic Sum[category], a box of its category, with cat.Sum's behaviour and monoidal.Sum's tensor and drawing
- [x] cat.Sum and monoidal.Sum as deprecated aliases, a test that the old names warn and are the new class
- [ ] every existing test passes; plan the remaining levels in the PR body
