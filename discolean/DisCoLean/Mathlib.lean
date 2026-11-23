import Mathlib.CategoryTheory.Category.Basic
import DisCoLean.ABC

/-!
# Compiled categories are categories of Mathlib

A `Category` of `discopy.abc` with its laws is a category of Mathlib on its
objects: the morphisms from `A` to `B` are those with this boundary, and the
laws of Mathlib are those compiled from `discopy.abc.Category`.
-/

namespace DisCoLean.ABC.Category

universe u v

variable {Ob : Type u} {Hom : Type v}

/-- The objects of a category, as the type a category of Mathlib is on. -/
structure Obj (C : Category Ob Hom) : Type u where
  ob : Ob

/-- A category with its laws, as a category of Mathlib. -/
def toMathlib (C : Category Ob Hom) (laws : C.Laws) :
    CategoryTheory.Category.{v} C.Obj where
  Hom A B := {f : Hom // C.dom f = A.ob ∧ C.cod f = B.ob}
  id A := ⟨C.id A.ob, C.ax A.ob⟩
  comp f g := ⟨C.comp f.1 g.1, C.cut _ _ _ f.1 g.1 f.2.1 f.2.2 g.2.1 g.2.2⟩
  id_comp f := Subtype.ext <| by
    obtain ⟨f, dom, cod⟩ := f
    simpa [dom] using (laws.unitality _ _ f dom cod).1
  comp_id f := Subtype.ext <| by
    obtain ⟨f, dom, cod⟩ := f
    simpa [cod] using (laws.unitality _ _ f dom cod).2.symm
  assoc f g h := Subtype.ext <| laws.associativity _ _ _ _ f.1 g.1 h.1
    f.2.1 f.2.2 g.2.1 g.2.2 h.2.1 h.2.2

end DisCoLean.ABC.Category
