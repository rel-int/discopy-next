`Category.typeparams` reads `ob` with no fallback: a monoid's objects are
`NoneType`, so `Nat.typeparams()` no longer gives `(Nat, Nat)` ([next#214](https://github.com/rel-int/discopy-next/pull/214)).
