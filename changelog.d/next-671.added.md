`abc.MonoidalCategory` states the laws of whiskering, `tensor_associativity`,
`tensor_unit`, `left_whiskering`, `right_whiskering` and `tensor_whiskering`,
those of a strict premonoidal category, `tensor_whiskering` broken on Markov
and Comarkov lenses and prisms, as is their unitality; `discopy.lean`
compiles a `Functor` of each class it compiles, its abstract methods and the objects it specialises its bases
to, e.g. `Nat` for `abc.PRO`, and `discolean/` proves the coherence of
`discopy.monoidal`:
Diagram(S) is the free model of the laws of `monoidal.Diagram` ([next#671](https://github.com/rel-int/discopy-next/pull/671)).
