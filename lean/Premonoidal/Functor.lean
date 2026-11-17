import Premonoidal.Diagram

/-!
# Diagram(S) is the free strict premonoidal category

A strict premonoidal functor out of Diagram(S) is given by where it sends
the generating objects and the boxes, and by nothing else.
-/

namespace Premonoidal

universe u v u' v'

/-- A strict premonoidal functor: it preserves domains, codomains,
identities, composition, the monoid of objects and both whiskerings,
all on the nose. -/
structure Functor (C : StrictPremonoidal.{u, v}) (D : StrictPremonoidal.{u', v'}) where
  ob : C.Ob → D.Ob
  map : C.Hom → D.Hom
  dom_map : ∀ f, D.dom (map f) = ob (C.dom f)
  cod_map : ∀ f, D.cod (map f) = ob (C.cod f)
  map_id : ∀ X, map (C.id X) = D.id (ob X)
  map_comp : ∀ f g, C.cod f = C.dom g → map (C.comp f g) = D.comp (map f) (map g)
  ob_unit : ob C.unit = D.unit
  ob_tensor : ∀ X Y, ob (C.tensor X Y) = D.tensor (ob X) (ob Y)
  map_whiskerLeft : ∀ X f, map (C.whiskerLeft X f) = D.whiskerLeft (ob X) (map f)
  map_whiskerRight : ∀ f X, map (C.whiskerRight f X) = D.whiskerRight (map f) (ob X)

theorem Functor.ext {C : StrictPremonoidal.{u, v}} {D : StrictPremonoidal.{u', v'}}
    {F G : Functor C D} (ob : F.ob = G.ob) (map : F.map = G.map) : F = G := by
  cases F; cases G; cases ob; cases map; rfl

/-- The product of a list of objects. -/
def StrictPremonoidal.prod (C : StrictPremonoidal.{u, v}) (Xs : List C.Ob) : C.Ob :=
  Xs.foldr C.tensor C.unit

namespace Lift

variable {S : Signature} (C : StrictPremonoidal.{u, v})
  (ob : S.Ob → C.Ob) (ar : S.Box → C.Hom)

/-- A list of generating objects goes to the product of their images. -/
def obs (X : List S.Ob) : C.Ob := C.prod (X.map ob)

/-- A layer goes to the image of its box, whiskered by those of its wires. -/
def layer (l : Layer S) : C.Hom :=
  C.whiskerLeft (obs C ob l.left) (C.whiskerRight (ar l.box) (obs C ob l.right))

/-- Layers starting from `X` go to the composite of their images. -/
def eval : List S.Ob → List (Layer S) → C.Hom
  | X, [] => C.id (obs C ob X)
  | _, l :: ls => C.comp (layer C ob ar l) (eval l.cod ls)

/-- A diagram goes to the composite of the images of its layers. -/
def map (f : Diagram S) : C.Hom := eval C ob ar f.dom f.layers

theorem obs_append (X Y : List S.Ob) :
    obs C ob (X ++ Y) = C.tensor (obs C ob X) (obs C ob Y) := by
  induction X with
  | nil => exact (C.unit_tensor _).symm
  | cons x X ih =>
    show C.tensor (ob x) (obs C ob (X ++ Y)) = C.tensor (C.tensor (ob x) (obs C ob X)) _
    rw [ih, C.tensor_assoc]

theorem obs_singleton (x : S.Ob) : obs C ob [x] = ob x := C.tensor_unit _

variable (hdom : ∀ b, C.dom (ar b) = obs C ob (S.dom b))
  (hcod : ∀ b, C.cod (ar b) = obs C ob (S.cod b))
include hdom hcod

omit hcod in
theorem dom_layer (l : Layer S) : C.dom (layer C ob ar l) = obs C ob l.dom := by
  simp only [layer, Layer.dom, C.dom_whiskerLeft, C.dom_whiskerRight, hdom,
    obs_append, C.tensor_assoc]

omit hdom in
theorem cod_layer (l : Layer S) : C.cod (layer C ob ar l) = obs C ob l.cod := by
  simp only [layer, Layer.cod, C.cod_whiskerLeft, C.cod_whiskerRight, hcod,
    obs_append, C.tensor_assoc]

theorem dom_eval {X : List S.Ob} {ls : List (Layer S)} (h : Composable X ls) :
    C.dom (eval C ob ar X ls) = obs C ob X ∧
      C.cod (eval C ob ar X ls) = obs C ob (codFrom X ls) := by
  induction ls generalizing X with
  | nil => exact ⟨C.dom_id _, C.cod_id _⟩
  | cons l ls ih =>
    have composable : C.cod (layer C ob ar l) = C.dom (eval C ob ar l.cod ls) := by
      rw [cod_layer C ob ar hcod, (ih h.2).1]
    exact ⟨by rw [eval, C.dom_comp _ _ composable, dom_layer C ob ar hdom, h.1],
      by rw [eval, C.cod_comp _ _ composable, (ih h.2).2]; rfl⟩

theorem eval_append {X : List S.Ob} {ls ms : List (Layer S)}
    (h : Composable X ls) (h' : Composable (codFrom X ls) ms) :
    eval C ob ar X (ls ++ ms)
      = C.comp (eval C ob ar X ls) (eval C ob ar (codFrom X ls) ms) := by
  induction ls generalizing X with
  | nil =>
    have unit := C.id_comp (eval C ob ar X ms)
    rw [(dom_eval C ob ar hdom hcod (X := X) h').1] at unit
    exact unit.symm
  | cons l ls ih =>
    have first : C.cod (layer C ob ar l) = C.dom (eval C ob ar l.cod ls) := by
      rw [cod_layer C ob ar hcod, (dom_eval C ob ar hdom hcod h.2).1]
    have second : C.cod (eval C ob ar l.cod ls)
        = C.dom (eval C ob ar (codFrom l.cod ls) ms) := by
      rw [(dom_eval C ob ar hdom hcod h.2).2,
        (dom_eval C ob ar hdom hcod (X := codFrom l.cod ls) h').1]
    show C.comp _ (eval C ob ar l.cod (ls ++ ms))
      = C.comp (C.comp _ _) (eval C ob ar (codFrom l.cod ls) ms)
    rw [ih h.2 h', C.assoc _ _ _ first second]

omit hdom hcod in
theorem layer_whiskerLeft (Z : List S.Ob) (l : Layer S) :
    layer C ob ar (Layer.whiskerLeft Z l)
      = C.whiskerLeft (obs C ob Z) (layer C ob ar l) := by
  simp only [layer, Layer.whiskerLeft, obs_append, C.tensor_whiskerLeft]

omit hdom hcod in
theorem layer_whiskerRight (l : Layer S) (Z : List S.Ob) :
    layer C ob ar (l.whiskerRight Z)
      = C.whiskerRight (layer C ob ar l) (obs C ob Z) := by
  simp only [layer, Layer.whiskerRight, obs_append, C.whiskerRight_tensor,
    C.whisker_assoc]

theorem eval_whiskerLeft (Z : List S.Ob) {X : List S.Ob} {ls : List (Layer S)}
    (h : Composable X ls) :
    eval C ob ar (Z ++ X) (ls.map (Layer.whiskerLeft Z))
      = C.whiskerLeft (obs C ob Z) (eval C ob ar X ls) := by
  induction ls generalizing X with
  | nil => simp only [List.map, eval, obs_append, C.whiskerLeft_id]
  | cons l ls ih =>
    have ih := ih h.2
    rw [← Layer.cod_whiskerLeft] at ih
    simp only [List.map, eval, ih, layer_whiskerLeft]
    rw [C.whiskerLeft_comp]
    rw [cod_layer C ob ar hcod, (dom_eval C ob ar hdom hcod h.2).1]

theorem eval_whiskerRight (Z : List S.Ob) {X : List S.Ob} {ls : List (Layer S)}
    (h : Composable X ls) :
    eval C ob ar (X ++ Z) (ls.map (Layer.whiskerRight · Z))
      = C.whiskerRight (eval C ob ar X ls) (obs C ob Z) := by
  induction ls generalizing X with
  | nil => simp only [List.map, eval, obs_append, C.id_whiskerRight]
  | cons l ls ih =>
    have ih := ih h.2
    rw [← Layer.cod_whiskerRight] at ih
    simp only [List.map, eval, ih, layer_whiskerRight]
    rw [C.comp_whiskerRight]
    rw [cod_layer C ob ar hcod, (dom_eval C ob ar hdom hcod h.2).1]

variable [DecidableEq S.Ob]

/-- The strict premonoidal functor out of Diagram(S) that extends `ob` and
`ar`, given that each box goes between the images of its domain and codomain. -/
def functor : Functor (Diagram.premonoidal S) C where
  ob := obs C ob
  map := map C ob ar
  dom_map f := (dom_eval C ob ar hdom hcod f.composable).1
  cod_map f := (dom_eval C ob ar hdom hcod f.composable).2
  map_id _ := rfl
  map_comp f g h := by
    have h' : codFrom f.dom f.layers = g.dom := h
    have append := eval_append C ob ar hdom hcod f.composable (ms := g.layers)
      (by rw [h']; exact g.composable)
    rw [h'] at append
    show eval C ob ar (f.comp g).dom (f.comp g).layers = _
    rw [Diagram.dom_comp h, Diagram.layers_comp h]
    exact append
  ob_unit := rfl
  ob_tensor := obs_append C ob
  map_whiskerLeft Z f := eval_whiskerLeft C ob ar hdom hcod Z f.composable
  map_whiskerRight f Z := eval_whiskerRight C ob ar hdom hcod Z f.composable

omit hdom [DecidableEq S.Ob] in
theorem map_box (b : S.Box) : map C ob ar (Diagram.box b) = ar b := by
  show C.comp (layer C ob ar ⟨[], b, []⟩) (C.id _) = _
  rw [← cod_layer C ob ar hcod, C.comp_id]
  exact (C.unit_whiskerLeft _).trans (C.whiskerRight_unit _)

end Lift

namespace Diagram

variable {S : Signature} [DecidableEq S.Ob]

/-- A diagram is its first layer then the rest, and a layer is its box
whiskered on both sides. -/
theorem cons_eq (X : List S.Ob) (l : Layer S) (ls : List (Layer S))
    (h : Composable X (l :: ls)) :
    (⟨X, l :: ls, h⟩ : Diagram S)
      = comp (whiskerLeft l.left (whiskerRight (box l.box) l.right)) ⟨l.cod, ls, h.2⟩ := by
  have composable : (whiskerLeft l.left (whiskerRight (box l.box) l.right)).cod
      = (⟨l.cod, ls, h.2⟩ : Diagram S).dom := by
    simp [cod, codFrom, whiskerLeft, whiskerRight, box, Layer.cod,
      Layer.whiskerLeft, Layer.whiskerRight]
  ext
  · rw [dom_comp composable]
    simp [whiskerLeft, whiskerRight, box, ← h.1, Layer.dom]
  · rw [layers_comp composable]
    simp [whiskerLeft, whiskerRight, box, Layer.whiskerLeft, Layer.whiskerRight]

end Diagram

/-- Two strict premonoidal functors out of Diagram(S) that agree on the
generating objects and on the boxes are equal. -/
theorem Functor.unique {S : Signature} [DecidableEq S.Ob]
    {C : StrictPremonoidal.{u, v}} (F G : Functor (Diagram.premonoidal S) C)
    (ob : ∀ x, F.ob [x] = G.ob [x])
    (ar : ∀ b, F.map (Diagram.box b) = G.map (Diagram.box b)) : F = G := by
  have obs : ∀ X, F.ob X = G.ob X := by
    intro X
    induction X with
    | nil => exact F.ob_unit.trans G.ob_unit.symm
    | cons x X ih =>
      have := F.ob_tensor [x] X
      have := G.ob_tensor [x] X
      simp_all [Diagram.premonoidal]
  refine Functor.ext (funext obs) (funext fun f => ?_)
  obtain ⟨X, ls, h⟩ := f
  induction ls generalizing X with
  | nil => exact (F.map_id X).trans ((congrArg C.id (obs X)).trans (G.map_id X).symm)
  | cons l ls ih =>
    rw [Diagram.cons_eq]
    have composable : (Diagram.premonoidal S).cod
        (Diagram.whiskerLeft l.left (Diagram.whiskerRight (Diagram.box l.box) l.right))
        = (Diagram.premonoidal S).dom ⟨l.cod, ls, h.2⟩ := by
      simp [Diagram.premonoidal, Diagram.cod, codFrom, Diagram.whiskerLeft,
        Diagram.whiskerRight, Diagram.box, Layer.cod, Layer.whiskerLeft,
        Layer.whiskerRight]
    have left := F.map_whiskerLeft l.left (Diagram.whiskerRight (Diagram.box l.box) l.right)
    have left' := G.map_whiskerLeft l.left (Diagram.whiskerRight (Diagram.box l.box) l.right)
    have right := F.map_whiskerRight (Diagram.box l.box) l.right
    have right' := G.map_whiskerRight (Diagram.box l.box) l.right
    have step := F.map_comp _ _ composable
    have step' := G.map_comp _ _ composable
    simp only [Diagram.premonoidal] at left left' right right' step step'
    rw [step, step', left, left', right, right', ar,
      show F.ob l.left = G.ob l.left from obs l.left,
      show F.ob l.right = G.ob l.right from obs l.right, ih l.cod h.2]

end Premonoidal
