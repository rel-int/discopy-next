import Premonoidal

/-!
# DisCoPy diagrams are the morphisms of the free premonoidal category

The model is `discopy.monoidal.Diagram` with one box per layer and no colours,
as `Diagram.encode` writes it. Morphisms are untyped, with `dom` and `cod` as
the Python has, so every law is a plain equality; a law about `f ≫ g` assumes
`cod f = dom g`. The definitions are restated by `rfl` and each theorem names
one broken implementation it rules out. `Audit.lean` covers this file, and
`test_premonoidal.py` links the model to the Python, not a proof.
-/

namespace Statements.Premonoidal
open _root_.Premonoidal

/-! ## Definitions: strict premonoidal categories -/

section
variable (C : StrictPremonoidal) (X Y Z : C.Ob) (f g h : C.Hom)

local infixl:65 " ≫ " => C.comp
local infixl:70 " ⊗ " => C.tensor
local infixr:75 " ⋊ " => C.whiskerLeft
local infixl:75 " ⋉ " => C.whiskerRight

/-- A category: identities and an associative composition. -/
theorem category_def (fg : C.cod f = C.dom g) (gh : C.cod g = C.dom h) :
    C.dom (C.id X) = X ∧ C.cod (C.id X) = X
    ∧ C.dom (f ≫ g) = C.dom f ∧ C.cod (f ≫ g) = C.cod g
    ∧ C.id (C.dom f) ≫ f = f ∧ f ≫ C.id (C.cod f) = f
    ∧ (f ≫ g) ≫ h = f ≫ (g ≫ h) :=
  ⟨C.dom_id X, C.cod_id X, C.dom_comp f g fg, C.cod_comp f g fg,
    C.id_comp f, C.comp_id f, C.assoc f g h fg gh⟩

/-- The objects are a monoid. -/
theorem monoid_def :
    C.unit ⊗ X = X ∧ X ⊗ C.unit = X ∧ (X ⊗ Y) ⊗ Z = X ⊗ (Y ⊗ Z) :=
  ⟨C.unit_tensor X, C.tensor_unit X, C.tensor_assoc X Y Z⟩

/-- Each object whiskers morphisms on the left and on the right, as a functor. -/
theorem whisker_def (fg : C.cod f = C.dom g) :
    C.dom (X ⋊ f) = X ⊗ C.dom f ∧ C.cod (X ⋊ f) = X ⊗ C.cod f
    ∧ C.dom (f ⋉ X) = C.dom f ⊗ X ∧ C.cod (f ⋉ X) = C.cod f ⊗ X
    ∧ X ⋊ C.id Y = C.id (X ⊗ Y) ∧ C.id X ⋉ Y = C.id (X ⊗ Y)
    ∧ X ⋊ (f ≫ g) = (X ⋊ f) ≫ (X ⋊ g) ∧ (f ≫ g) ⋉ X = (f ⋉ X) ≫ (g ⋉ X) :=
  ⟨C.dom_whiskerLeft X f, C.cod_whiskerLeft X f, C.dom_whiskerRight f X,
    C.cod_whiskerRight f X, C.whiskerLeft_id X Y, C.id_whiskerRight X Y,
    C.whiskerLeft_comp X f g fg, C.comp_whiskerRight f g X fg⟩

/-- Whiskering is strictly unital and associative. No interchange is asked. -/
theorem strict_def :
    C.unit ⋊ f = f ∧ f ⋉ C.unit = f
    ∧ (X ⊗ Y) ⋊ f = X ⋊ (Y ⋊ f) ∧ f ⋉ (X ⊗ Y) = (f ⋉ X) ⋉ Y
    ∧ (X ⋊ f) ⋉ Y = X ⋊ (f ⋉ Y) :=
  ⟨C.unit_whiskerLeft f, C.whiskerRight_unit f, C.tensor_whiskerLeft X Y f,
    C.whiskerRight_tensor f X Y, C.whisker_assoc X f Y⟩

/-- The tensor of two morphisms does the first one first. -/
theorem tensorHom_def : C.tensorHom f g = (f ⋉ C.dom g) ≫ (C.cod f ⋊ g) := rfl

/-- A strict premonoidal functor preserves all of it, on the nose. -/
theorem functor_def (D : StrictPremonoidal) (F : Functor C D) (fg : C.cod f = C.dom g) :
    D.dom (F.map f) = F.ob (C.dom f) ∧ D.cod (F.map f) = F.ob (C.cod f)
    ∧ F.map (C.id X) = D.id (F.ob X) ∧ F.map (f ≫ g) = D.comp (F.map f) (F.map g)
    ∧ F.ob C.unit = D.unit ∧ F.ob (X ⊗ Y) = D.tensor (F.ob X) (F.ob Y)
    ∧ F.map (X ⋊ f) = D.whiskerLeft (F.ob X) (F.map f)
    ∧ F.map (f ⋉ X) = D.whiskerRight (F.map f) (F.ob X) :=
  ⟨F.dom_map f, F.cod_map f, F.map_id X, F.map_comp f g fg, F.ob_unit,
    F.ob_tensor X Y, F.map_whiskerLeft X f, F.map_whiskerRight f X⟩

/-- The product of a list of objects. -/
theorem prod_def (Xs : List C.Ob) : C.prod [] = C.unit
    ∧ C.prod (X :: Xs) = X ⊗ C.prod Xs := ⟨rfl, rfl⟩

end

/-! ## Definitions: Diagram(S), for a signature `S` of objects and boxes -/

variable {S : Signature}

/-- A layer is a box between wires. -/
theorem layer_def (l : Layer S) : l.dom = l.left ++ S.dom l.box ++ l.right
    ∧ l.cod = l.left ++ S.cod l.box ++ l.right := ⟨rfl, rfl⟩

/-- A diagram is a domain and a list of layers, each starting where the
previous one ends. -/
theorem diagram_def (X : List S.Ob) (l : Layer S) (ls : List (Layer S)) :
    (Composable X [] ↔ True) ∧ codFrom X [] = X
    ∧ (Composable X (l :: ls) ↔ l.dom = X ∧ Composable l.cod ls)
    ∧ codFrom X (l :: ls) = codFrom l.cod ls
    ∧ ∀ f : Diagram S, Composable f.dom f.layers ∧ f.cod = codFrom f.dom f.layers :=
  ⟨Iff.rfl, rfl, Iff.rfl, rfl, fun f => ⟨f.composable, rfl⟩⟩

variable [DecidableEq S.Ob]

/-- The identity is empty, composition concatenates, whiskering adds wires
to every layer, a box is one layer. -/
theorem operations_def (X : List S.Ob) (f g : Diagram S) (b : S.Box) :
    (Diagram.id X).dom = X ∧ (Diagram.id X).layers = []
    ∧ (f.cod = g.dom → (f.comp g).dom = f.dom
      ∧ (f.comp g).layers = f.layers ++ g.layers)
    ∧ (Diagram.whiskerLeft X f).dom = X ++ f.dom ∧ (Diagram.whiskerLeft X f).layers
      = f.layers.map (fun l => ⟨X ++ l.left, l.box, l.right⟩)
    ∧ (f.whiskerRight X).dom = f.dom ++ X ∧ (f.whiskerRight X).layers
      = f.layers.map (fun l => ⟨l.left, l.box, l.right ++ X⟩)
    ∧ (Diagram.box b).dom = S.dom b ∧ (Diagram.box b).layers = [⟨[], b, []⟩] :=
  ⟨rfl, rfl, fun h => ⟨Diagram.dom_comp h, Diagram.layers_comp h⟩,
    rfl, rfl, rfl, rfl, rfl, rfl⟩

/-! ## Theorems -/

/-- Diagram(S) is a strict premonoidal category, with these operations.
Rules out: whiskering that drops the wires of a layer, so that `Z ⋊ f` and
`f` end on different objects. -/
theorem premonoidal : let C := Diagram.premonoidal S
    C.Ob = List S.Ob ∧ C.Hom = Diagram S ∧ C.unit = ([] : List S.Ob)
    ∧ C.tensor = (List.append : List S.Ob → _) ∧ C.dom = (Diagram.dom : Diagram S → _)
    ∧ C.cod = (Diagram.cod : Diagram S → _) ∧ C.id = (Diagram.id : _ → Diagram S)
    ∧ C.comp = (Diagram.comp : Diagram S → _)
    ∧ C.whiskerLeft = (Diagram.whiskerLeft : _ → Diagram S → _)
    ∧ C.whiskerRight = (Diagram.whiskerRight : Diagram S → _) :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The tensor of diagrams is `f ⋉ dom g` then `cod f ⋊ g`, as `Diagram.tensor`.
Rules out: a tensor that does the second diagram first. -/
theorem tensor_def (f g : Diagram S) : f.tensor g
    = (f.whiskerRight g.dom).comp (Diagram.whiskerLeft f.cod g) := rfl

/-- Interchange fails for boxes: the two orders of `f ⊗ g` are equal only
when `f` and `g` are one and the same scalar.
Rules out: a model that quotients diagrams by interchange. -/
theorem not_monoidal (f g : S.Box) :
    let (f', g') := (Diagram.box f, Diagram.box g)
    (f'.whiskerRight (S.dom g)).comp (Diagram.whiskerLeft (S.cod f) g')
      = (Diagram.whiskerLeft (S.dom f) g').comp (f'.whiskerRight (S.cod g))
    ↔ f = g ∧ S.dom f = [] ∧ S.cod f = [] := Diagram.interchange_iff f g

/-- Diagram(S) is free: objects of `C` for the generating objects, and for
each box a morphism between the products of its domain and codomain, extend
to a strict premonoidal functor.
Rules out: a functor that whiskers the image of a box on the wrong side. -/
theorem lift (C : StrictPremonoidal) (ob : S.Ob → C.Ob) (ar : S.Box → C.Hom)
    (dom : ∀ b, C.dom (ar b) = C.prod ((S.dom b).map ob))
    (cod : ∀ b, C.cod (ar b) = C.prod ((S.cod b).map ob)) :
    ∃ F : Functor (Diagram.premonoidal S) C,
      (∀ x, F.ob [x] = ob x) ∧ ∀ b, F.map (Diagram.box b) = ar b :=
  ⟨Lift.functor C ob ar dom cod, Lift.obs_singleton C ob, Lift.map_box C ob ar cod⟩

/-- The extension is unique: strict premonoidal functors from Diagram(S)
that agree on the generating objects and on the boxes are equal.
Rules out: a diagram that is not a composite of whiskered boxes. -/
theorem lift_unique (C : StrictPremonoidal) (F G : Functor (Diagram.premonoidal S) C)
    (ob : ∀ x, F.ob [x] = G.ob [x])
    (ar : ∀ b, F.map (Diagram.box b) = G.map (Diagram.box b)) : F = G :=
  Functor.unique F G ob ar

end Statements.Premonoidal
