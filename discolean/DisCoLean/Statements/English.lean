import DisCoLean.Statements.Premonoidal

/-!
# DisCoPy diagrams are the arrows of the free premonoidal category

The statement is the one the abstract categorial grammar of
`discopy.lambek.abstract` prints for that sentence, word for word, and
`test/lambek/abstract.py` checks that it is. The sentence with *monoidal*
prints a statement about a `StrictMonoidal` this model does not have, and
`Statements.Premonoidal.not_monoidal` says why it cannot.
-/

namespace Statements.English
open _root_.Premonoidal Statements.Premonoidal

/-- DisCoPy diagrams are the arrows of the free premonoidal category. -/
theorem diagrams_are_arrows {S : Signature} [DecidableEq S.Ob] :
    (Diagram.premonoidal S).Hom = Diagram S
      ∧ (∀ (C : StrictPremonoidal) (ob : S.Ob → C.Ob) (ar : S.Box → C.Hom),
        (∀ b, C.dom (ar b) = C.prod ((S.dom b).map ob)) →
        (∀ b, C.cod (ar b) = C.prod ((S.cod b).map ob)) →
        ∃ F : Functor (Diagram.premonoidal S) C,
          (∀ x, F.ob [x] = ob x) ∧ ∀ b, F.map (Diagram.box b) = ar b)
      ∧ ∀ (C : StrictPremonoidal) (F G : Functor (Diagram.premonoidal S) C),
        (∀ x, F.ob [x] = G.ob [x]) →
        (∀ b, F.map (Diagram.box b) = G.map (Diagram.box b)) → F = G
    := ⟨rfl, lift, lift_unique⟩

end Statements.English
