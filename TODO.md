> Implement calls 1 and 2 of rel-int/discopy-next#485, as USER (toumix) ruled live 11-09: '1. `BraidedCategory.braid_inverse`, defaulting to `braid(...).dagger()`. 2. `dagger_monoidality` moves to the first class with both a dagger and a tensor; declarations that exist only for the misplacement go.' Call 3 (re-running the 26 XPASS cells under explore) is not part of this PR.

- [x] `BraidedCategory.braid_inverse`, defaulting to `braid(...).dagger()`, and the laws and the rule that read the inverse braid as a dagger restated with it
- [x] `DaggerMonoidalCategory(DaggerCategory, MonoidalCategory)` carrying `dagger_monoidality`, inherited by every category with both, and the declarations that only existed for the misplacement dropped
- [ ] the property matrix before and after, with `--proptest-baseline`, every lost cell explained
- [ ] tests, docs and a changelog bullet
