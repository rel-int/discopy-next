`discopy.sum` goes, and `Sum` lives in each module again: `cat.Sum` is a box
of formal sums with their composition, `monoidal.Sum` adds the tensor and
each level's `Sum` subclasses its parent's: import `Sum` from the module of
its category, e.g. `monoidal.Sum` for `sum.Sum[monoidal.Diagram]`.
`Sum[C]` and the `subscript_bases` hook of `NamedGeneric` go with it
([next#736](https://github.com/rel-int/discopy-next/pull/736)).
