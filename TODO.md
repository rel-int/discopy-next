> next#373: Declare it — Drawing declares DaggerCategory, additive.Function's swap-only dagger is renamed (inverse of a swap). Predicate becomes issubclass(c, DaggerCategory) everywhere.

- [x] Drawing declares DaggerCategory, its laws checked
- [x] additive.Function.dagger renamed inverse, callers updated
- [x] cat.Functor's dagger predicate is issubclass(cod, DaggerCategory)
- [x] tests, the issue's reproducer included
- [x] CHANGELOG bullet
- [ ] ask USER: interaction.Diagram, finset.Cospan and hopf.Intertwiner have a total dagger and declare no DaggerCategory, so a functor into them now reads a daggered box off ar_map
