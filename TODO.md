> write the brief for the symmetric slice
>
> yes ofc Stream and Drawing should be SymmetricCategory
>
> (toumix, live, 2026-11-12. The brief is https://github.com/rel-int/discopy-next/issues/513, option A picked.)

- [ ] `symmetric.Functor.__call__` in two cases, by the level of the codomain: `cod.from_permutation` for a `SymmetricCategory`, swap by swap otherwise
- [ ] `stream.Stream` and `drawing.Drawing` subclass `SymmetricCategory`; the property matrix's new cells listed with their outcome
- [ ] a new test that fails on `main`: a `SymmetricCategory` defining only `swap` receives a three-cycle through `from_permutation`; the five tests named in #513 pass unedited
- [ ] CHANGELOG `Changed`, every check of CONTRIBUTING.md
