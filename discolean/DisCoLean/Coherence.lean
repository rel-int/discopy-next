import DisCoLean.ABC
import DisCoLean.Premonoidal.Diagram

/-!
# The coherence of `discopy.monoidal`

`DisCoLean.ABC.monoidal` is the theory of `monoidal.Diagram`: the laws of
`abc.MonoidalCategory` but bifunctoriality, which the Python declares broken.
A strict premonoidal category is a model of it, with the tensor of two
morphisms the first one first, and so is Diagram(S), the free strict
premonoidal category of `Premonoidal/Functor.lean`.

The converse does not hold: the theory has no law of whiskering, e.g.
`(f ⊗ g) ⊗ h = f ⊗ (g ⊗ h)`, since `abc.MonoidalCategory` states none, so its
free model is not Diagram(S) until it does.
-/

namespace Premonoidal.StrictPremonoidal

/-- A strict premonoidal category as an `abc.MonoidalCategory`, the tensor
of morphisms `f ⋉ dom g` then `cod f ⋊ g`. -/
def toABC (C : StrictPremonoidal) : DisCoLean.ABC.MonoidalCategory C.Ob C.Hom where
  dom := C.dom
  cod := C.cod
  id := C.id
  comp := C.comp
  ax X := ⟨C.dom_id X, C.cod_id X⟩
  cut _ _ _ f g dom_f cod_f dom_g cod_g :=
    ⟨(C.dom_comp f g (cod_f.trans dom_g.symm)).trans dom_f,
     (C.cod_comp f g (cod_f.trans dom_g.symm)).trans cod_g⟩
  tensor := C.tensorHom
  tensorOb := C.tensor
  mix _ _ _ _ f g dom_f cod_f dom_g cod_g := by
    have h : C.cod (C.whiskerRight f (C.dom g))
        = C.dom (C.whiskerLeft (C.cod f) g) := by
      rw [C.cod_whiskerRight, C.dom_whiskerLeft]
    refine ⟨?_, ?_⟩ <;> simp only [tensorHom]
    · rw [C.dom_comp _ _ h, C.dom_whiskerRight, dom_f, dom_g]
    · rw [C.cod_comp _ _ h, C.cod_whiskerLeft, cod_f, cod_g]

/-- A strict premonoidal category has the laws of `monoidal.Diagram`. -/
theorem toABC_monoidal (C : StrictPremonoidal) : DisCoLean.ABC.monoidal C.toABC := by
  refine ⟨?_, ?_, ?_⟩
  · exact fun _ _ f _ _ => ⟨C.id_comp f, (C.comp_id f).symm⟩
  · exact fun _ _ _ _ f g h _ cod_f dom_g cod_g dom_h _ =>
      C.assoc f g h (cod_f.trans dom_g.symm) (cod_g.trans dom_h.symm)
  · intro X Y
    show C.tensorHom (C.id X) (C.id Y) = C.id (C.tensor X Y)
    have h := C.id_comp (C.id (C.tensor X Y))
    rw [C.dom_id] at h
    simp only [tensorHom, C.dom_id, C.cod_id, C.id_whiskerRight, C.whiskerLeft_id, h]

end Premonoidal.StrictPremonoidal

namespace Premonoidal.Diagram

/-- Diagram(S) has the laws of `monoidal.Diagram`. -/
theorem monoidal (S : Signature) [DecidableEq S.Ob] :
    DisCoLean.ABC.monoidal (premonoidal S).toABC :=
  (premonoidal S).toABC_monoidal

end Premonoidal.Diagram
