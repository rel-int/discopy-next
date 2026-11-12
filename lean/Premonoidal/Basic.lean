/-!
# Categories and strict premonoidal categories

Morphisms are untyped, as in the Python: one type `Hom` with `dom` and `cod`
into the objects, so that the laws relating `(X ⊗ Y) ⊗ Z` and `X ⊗ (Y ⊗ Z)`
are plain equalities of morphisms, with no transport along an equality of
objects. Composition is a total function, written in diagrammatic order, and
every law about it assumes its arguments composable: nothing is said of
`comp f g` when `cod f ≠ dom g`, where the Python raises.
-/

namespace Premonoidal

universe u v

/-- A category, with `comp f g` meaning `f` then `g`. -/
structure Category where
  Ob : Type u
  Hom : Type v
  dom : Hom → Ob
  cod : Hom → Ob
  id : Ob → Hom
  comp : Hom → Hom → Hom
  dom_id : ∀ X, dom (id X) = X
  cod_id : ∀ X, cod (id X) = X
  dom_comp : ∀ f g, cod f = dom g → dom (comp f g) = dom f
  cod_comp : ∀ f g, cod f = dom g → cod (comp f g) = cod g
  id_comp : ∀ f, comp (id (dom f)) f = f
  comp_id : ∀ f, comp f (id (cod f)) = f
  assoc : ∀ f g h, cod f = dom g → cod g = dom h →
    comp (comp f g) h = comp f (comp g h)

/-- A strict premonoidal category (Power and Robinson, 1997): the objects are
a monoid, each object whiskers morphisms on either side functorially, and the
whiskerings are strictly unital, associative and commute with each other.
No interchange law is asked. Equivalently, a one-object sesquicategory. -/
structure StrictPremonoidal extends Category.{u, v} where
  unit : Ob
  tensor : Ob → Ob → Ob
  unit_tensor : ∀ X, tensor unit X = X
  tensor_unit : ∀ X, tensor X unit = X
  tensor_assoc : ∀ X Y Z, tensor (tensor X Y) Z = tensor X (tensor Y Z)
  whiskerLeft : Ob → Hom → Hom
  whiskerRight : Hom → Ob → Hom
  dom_whiskerLeft : ∀ X f, dom (whiskerLeft X f) = tensor X (dom f)
  cod_whiskerLeft : ∀ X f, cod (whiskerLeft X f) = tensor X (cod f)
  dom_whiskerRight : ∀ f X, dom (whiskerRight f X) = tensor (dom f) X
  cod_whiskerRight : ∀ f X, cod (whiskerRight f X) = tensor (cod f) X
  whiskerLeft_id : ∀ X Y, whiskerLeft X (id Y) = id (tensor X Y)
  id_whiskerRight : ∀ X Y, whiskerRight (id X) Y = id (tensor X Y)
  whiskerLeft_comp : ∀ X f g, cod f = dom g →
    whiskerLeft X (comp f g) = comp (whiskerLeft X f) (whiskerLeft X g)
  comp_whiskerRight : ∀ f g X, cod f = dom g →
    whiskerRight (comp f g) X = comp (whiskerRight f X) (whiskerRight g X)
  unit_whiskerLeft : ∀ f, whiskerLeft unit f = f
  whiskerRight_unit : ∀ f, whiskerRight f unit = f
  tensor_whiskerLeft : ∀ X Y f,
    whiskerLeft (tensor X Y) f = whiskerLeft X (whiskerLeft Y f)
  whiskerRight_tensor : ∀ f X Y,
    whiskerRight f (tensor X Y) = whiskerRight (whiskerRight f X) Y
  whisker_assoc : ∀ X f Y,
    whiskerRight (whiskerLeft X f) Y = whiskerLeft X (whiskerRight f Y)

/-- The tensor of two morphisms, the first one first: `f ⊗ g` is
`f ⋉ dom g` then `cod f ⋊ g`. -/
def StrictPremonoidal.tensorHom (C : StrictPremonoidal) (f g : C.Hom) : C.Hom :=
  C.comp (C.whiskerRight f (C.dom g)) (C.whiskerLeft (C.cod f) g)

end Premonoidal
