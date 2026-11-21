> toumix, in the Amp thread of this pull request (https://ampcode.com/threads/T-0729b388-26a3-4a3d-ac11-e727fdb06d74), on weakening `str_transparency` to an equation between terms of any level:
>
> > doesn't look very good, it would be better to introduce a Ty subclass for each module with its own pointer to the Term class

- [x] a Ty subclass in each module that shares one (spacial, braided, balanced, symmetric, comarkov, markov), the ob of its Diagram
- [x] each Ty points to the terms of its module: a type applied to a function is a kappa of its own level, and str_transparency is an equation between terms of that level again
- [x] the modules from biclosed up: their Ty subclasses the one of each level their Diagram subclasses, filed as #683
- [WIP] @T-0729b388-26a3-4a3d-ac11-e727fdb06d74-2026-11-21 11:41 tests, docs and changelog; the term cells of the matrix under the explore profile
