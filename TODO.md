make a new discopy-next branch (no PR) with all the upstream fixes we want to make. if not already the case, the constructors in `testable` should map to rules instantiations. feel free to make any other changes as well

- [ ] constructors of `testable` (`Tensor`, `Unit`, `L`, `R`, `Delay`, `Bang`, `Over`, `Under`, `Repeat`) as instances of type-formation rules, so that a category's objects are derivable like its morphisms
- [ ] report or fix `comarkov.Diagram.hypergraph_section` on `Trace(Merge(x, 2), left=True)` (raises: the only spiders of comarkov.Diagram are its merges)
- [ ] changelog entry, docs and tests for each change
- [ ] every check of CONTRIBUTING.md's "Run the unit tests"
