`abc.LinearCategory`, a linear category (Bierman, TLCA 1995) with neither
symmetry nor closure: a monoidal comonad `bang` with `dereliction` and
`digging`, lax monoidal with a `laxator`, each `!x` a comonoid by
`contraction` and `weakening`, and an invertible `exchange` of `!x` with
any object, its 28 laws as axioms.
`discopy.linear.monoidal` is the free one, beside `monoidal`: `BangOb`,
the wire holding a type inside `x.bang()`, `Bang`, the bubble `f.bang()`,
and the boxes `Dereliction`, `Digging`, `Contraction`, `Weakening`,
`Exchange` and `Laxator`. A `Functor` sends them through the `bang` and the
structure of its codomain when it is a linear category, and through its
`ob_map` and `ar_map` otherwise, `f.bang()` as a box of its own.
`Diagram.copy` and `Diagram.discard` copy a banged wire by contraction
and discard it by weakening, and raise on any other type, and
`Diagram.wiring` copies and exchanges banged wires. `linear.monoidal.TermBase`,
`Variable`, `Constant`, `Application` and `Kappa` are the kappa terms of
`monoidal` where a banged variable is shared by contraction and, unused,
discarded by weakening, a `linear.monoidal.Ty` applied to a function is a
`linear.monoidal.Kappa` through its `term_factory`, with a box or without, and a box
called on terms is its constant applied to them, e.g. `f.bang()(x)`. The laws are declared failing on the free category. `optyx.Term`, in
`optyx.term`, is a diagram of it on the dual-rail `qubit`, written with
`f.bang()` in the function syntax or as a kappa term of `linear.monoidal`, and
`Term.to_channel` compiles it to an optical protocol, `!` as Fock space,
but for `Digging`, `Laxator` and boxes on more than one qubit:
a bang to the second quantisation of its unitary, the conjugate transpose
for a dagger, `Contraction` to the W node on each mode, `Weakening` to the
vacuum and `Dereliction` to heralding one photon. `optyx.term` has kappa
terms of its own, so a term with no box is an `optyx.Term` too. The notebook
`optyx/linear_logic` compiles `copy(!H(x))` with it to a Bell state
([next#571](https://github.com/rel-int/discopy-next/pull/571), [next#686](https://github.com/rel-int/discopy-next/pull/686)).
