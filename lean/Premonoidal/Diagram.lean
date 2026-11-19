import Premonoidal.Basic

/-!
# Diagram(S), the diagrams of a monoidal signature

The model of `discopy.monoidal.Diagram` restricted to one box per layer and
no colours: a domain and a composable list of layers `(left, box, right)`,
which is what `Diagram.encode` writes with `len(left)` as offset.
-/

namespace Premonoidal

/-- A monoidal signature: generating objects, boxes, and for each box a
domain and a codomain, lists of generating objects. -/
structure Signature where
  Ob : Type
  Box : Type
  dom : Box → List Ob
  cod : Box → List Ob

variable {S : Signature}

/-- A box with wires on its left and on its right. -/
structure Layer (S : Signature) where
  left : List S.Ob
  box : S.Box
  right : List S.Ob

namespace Layer

def dom (l : Layer S) : List S.Ob := l.left ++ S.dom l.box ++ l.right
def cod (l : Layer S) : List S.Ob := l.left ++ S.cod l.box ++ l.right

def whiskerLeft (Z : List S.Ob) (l : Layer S) : Layer S :=
  { l with left := Z ++ l.left }

def whiskerRight (l : Layer S) (Z : List S.Ob) : Layer S :=
  { l with right := l.right ++ Z }

@[simp] theorem dom_whiskerLeft (Z : List S.Ob) (l : Layer S) :
    (whiskerLeft Z l).dom = Z ++ l.dom := by simp [whiskerLeft, dom]

@[simp] theorem cod_whiskerLeft (Z : List S.Ob) (l : Layer S) :
    (whiskerLeft Z l).cod = Z ++ l.cod := by simp [whiskerLeft, cod]

@[simp] theorem dom_whiskerRight (l : Layer S) (Z : List S.Ob) :
    (whiskerRight l Z).dom = l.dom ++ Z := by simp [whiskerRight, dom]

@[simp] theorem cod_whiskerRight (l : Layer S) (Z : List S.Ob) :
    (whiskerRight l Z).cod = l.cod ++ Z := by simp [whiskerRight, cod]

end Layer

/-- The layers compose, starting from `X`. -/
def Composable : List S.Ob → List (Layer S) → Prop
  | _, [] => True
  | X, l :: ls => l.dom = X ∧ Composable l.cod ls

/-- Where the layers end, starting from `X`. -/
def codFrom : List S.Ob → List (Layer S) → List S.Ob
  | X, [] => X
  | _, l :: ls => codFrom l.cod ls

theorem codFrom_append (X : List S.Ob) (ls ms : List (Layer S)) :
    codFrom X (ls ++ ms) = codFrom (codFrom X ls) ms := by
  induction ls generalizing X with
  | nil => rfl
  | cons l ls ih => exact ih l.cod

theorem composable_append {X : List S.Ob} {ls ms : List (Layer S)} :
    Composable X (ls ++ ms) ↔ Composable X ls ∧ Composable (codFrom X ls) ms := by
  induction ls generalizing X with
  | nil => simp [Composable, codFrom]
  | cons l ls ih => simp [Composable, codFrom, ih, and_assoc]

theorem composable_whiskerLeft (Z : List S.Ob) {X : List S.Ob}
    {ls : List (Layer S)} (h : Composable X ls) :
    Composable (Z ++ X) (ls.map (Layer.whiskerLeft Z)) := by
  induction ls generalizing X with
  | nil => trivial
  | cons l ls ih => exact ⟨by simp [h.1], by simpa using ih h.2⟩

theorem composable_whiskerRight (Z : List S.Ob) {X : List S.Ob}
    {ls : List (Layer S)} (h : Composable X ls) :
    Composable (X ++ Z) (ls.map (Layer.whiskerRight · Z)) := by
  induction ls generalizing X with
  | nil => trivial
  | cons l ls ih => exact ⟨by simp [h.1], by simpa using ih h.2⟩

theorem codFrom_whiskerLeft (Z X : List S.Ob) (ls : List (Layer S)) :
    codFrom (Z ++ X) (ls.map (Layer.whiskerLeft Z)) = Z ++ codFrom X ls := by
  induction ls generalizing X with
  | nil => rfl
  | cons l ls ih => simpa [codFrom] using ih l.cod

theorem codFrom_whiskerRight (Z X : List S.Ob) (ls : List (Layer S)) :
    codFrom (X ++ Z) (ls.map (Layer.whiskerRight · Z)) = codFrom X ls ++ Z := by
  induction ls generalizing X with
  | nil => rfl
  | cons l ls ih => simpa [codFrom] using ih l.cod

/-- A diagram: a domain and a composable list of layers. -/
structure Diagram (S : Signature) where
  dom : List S.Ob
  layers : List (Layer S)
  composable : Composable dom layers

namespace Diagram

@[ext] theorem ext {f g : Diagram S} (dom : f.dom = g.dom)
    (layers : f.layers = g.layers) : f = g := by
  cases f; cases g; cases dom; cases layers; rfl

def cod (f : Diagram S) : List S.Ob := codFrom f.dom f.layers

/-- The identity is the empty list of layers. -/
def id (X : List S.Ob) : Diagram S := ⟨X, [], trivial⟩

/-- Composition is concatenation; a diagram that does not compose with the
next is left as it is, where the Python raises. -/
def comp [DecidableEq S.Ob] (f g : Diagram S) : Diagram S :=
  if h : f.cod = g.dom then
    ⟨f.dom, f.layers ++ g.layers,
      composable_append.2 ⟨f.composable, by rw [← cod, h]; exact g.composable⟩⟩
  else f

/-- `Z ⋊ f` adds `Z` to the left of every layer. -/
def whiskerLeft (Z : List S.Ob) (f : Diagram S) : Diagram S :=
  ⟨Z ++ f.dom, f.layers.map (Layer.whiskerLeft Z),
    composable_whiskerLeft Z f.composable⟩

/-- `f ⋉ Z` adds `Z` to the right of every layer. -/
def whiskerRight (f : Diagram S) (Z : List S.Ob) : Diagram S :=
  ⟨f.dom ++ Z, f.layers.map (Layer.whiskerRight · Z),
    composable_whiskerRight Z f.composable⟩

/-- A box as a diagram of one layer. -/
def box (b : S.Box) : Diagram S :=
  ⟨S.dom b, [⟨[], b, []⟩], by simp [Composable, Layer.dom]⟩

theorem cod_whiskerLeft (Z : List S.Ob) (f : Diagram S) :
    (whiskerLeft Z f).cod = Z ++ f.cod := codFrom_whiskerLeft Z f.dom f.layers

theorem cod_whiskerRight (f : Diagram S) (Z : List S.Ob) :
    (whiskerRight f Z).cod = f.cod ++ Z := codFrom_whiskerRight Z f.dom f.layers

variable [DecidableEq S.Ob]

theorem dom_comp {f g : Diagram S} (h : f.cod = g.dom) :
    (comp f g).dom = f.dom := by simp [comp, h]

theorem layers_comp {f g : Diagram S} (h : f.cod = g.dom) :
    (comp f g).layers = f.layers ++ g.layers := by simp [comp, h]

theorem cod_comp {f g : Diagram S} (h : f.cod = g.dom) :
    (comp f g).cod = g.cod := by
  simp only [cod, dom_comp h, layers_comp h, codFrom_append]
  exact congrArg (codFrom · g.layers) h

theorem whiskerLeft_comp (Z : List S.Ob) {f g : Diagram S} (h : f.cod = g.dom) :
    whiskerLeft Z (comp f g) = comp (whiskerLeft Z f) (whiskerLeft Z g) := by
  have h' : (whiskerLeft Z f).cod = (whiskerLeft Z g).dom := by
    rw [cod_whiskerLeft, h]; rfl
  ext
  · rw [dom_comp h']; simp [whiskerLeft, dom_comp h]
  · rw [layers_comp h']; simp [whiskerLeft, layers_comp h]

theorem comp_whiskerRight {f g : Diagram S} (Z : List S.Ob) (h : f.cod = g.dom) :
    whiskerRight (comp f g) Z = comp (whiskerRight f Z) (whiskerRight g Z) := by
  have h' : (whiskerRight f Z).cod = (whiskerRight g Z).dom := by
    rw [cod_whiskerRight, h]; rfl
  ext
  · rw [dom_comp h']; simp [whiskerRight, dom_comp h]
  · rw [layers_comp h']; simp [whiskerRight, layers_comp h]

/-- Diagram(S) is a strict premonoidal category: objects are lists of
generating objects under concatenation. -/
def premonoidal (S : Signature) [DecidableEq S.Ob] : StrictPremonoidal where
  Ob := List S.Ob
  Hom := Diagram S
  dom := dom
  cod := cod
  id := id
  comp := comp
  dom_id _ := rfl
  cod_id _ := rfl
  dom_comp _ _ := dom_comp
  cod_comp _ _ := cod_comp
  id_comp f := by
    have h : (id f.dom).cod = f.dom := rfl
    ext
    · rw [dom_comp h]; rfl
    · rw [layers_comp h]; rfl
  comp_id f := by
    have h : f.cod = (id f.cod).dom := rfl
    ext
    · rw [dom_comp h]
    · rw [layers_comp h]; simp [id]
  assoc f g k fg gk := by
    have fg' : f.cod = (comp g k).dom := by rw [dom_comp gk]; exact fg
    have gk' : (comp f g).cod = k.dom := by rw [cod_comp fg]; exact gk
    ext
    · rw [dom_comp gk', dom_comp fg, dom_comp fg']
    · rw [layers_comp gk', layers_comp fg, layers_comp fg', layers_comp gk,
        List.append_assoc]
  unit := []
  tensor := (· ++ ·)
  unit_tensor := List.nil_append
  tensor_unit := List.append_nil
  tensor_assoc := List.append_assoc
  whiskerLeft := whiskerLeft
  whiskerRight := whiskerRight
  dom_whiskerLeft _ _ := rfl
  cod_whiskerLeft := cod_whiskerLeft
  dom_whiskerRight _ _ := rfl
  cod_whiskerRight := cod_whiskerRight
  whiskerLeft_id _ _ := rfl
  id_whiskerRight _ _ := rfl
  whiskerLeft_comp Z _ _ := whiskerLeft_comp Z
  comp_whiskerRight _ _ Z := comp_whiskerRight Z
  unit_whiskerLeft f := by
    ext <;> simp [whiskerLeft, Layer.whiskerLeft]
  whiskerRight_unit f := by
    ext <;> simp [whiskerRight, Layer.whiskerRight]
  tensor_whiskerLeft X Y f := by
    ext <;> simp [whiskerLeft, Layer.whiskerLeft]
  whiskerRight_tensor f X Y := by
    ext <;> simp [whiskerRight, Layer.whiskerRight]
  whisker_assoc X f Y := by
    ext <;> simp [whiskerLeft, whiskerRight, Layer.whiskerLeft,
      Layer.whiskerRight]

/-- The tensor of two diagrams, the first one first. -/
def tensor (f g : Diagram S) : Diagram S := (premonoidal S).tensorHom f g

/-- Interchange holds for two boxes only when they are one scalar. -/
theorem interchange_iff (f g : S.Box) :
    comp (whiskerRight (box f) (S.dom g)) (whiskerLeft (S.cod f) (box g))
      = comp (whiskerLeft (S.dom f) (box g)) (whiskerRight (box f) (S.cod g))
    ↔ f = g ∧ S.dom f = [] ∧ S.cod f = [] := by
  have left : (whiskerRight (box f) (S.dom g)).cod
      = (whiskerLeft (S.cod f) (box g)).dom := by
    simp [cod, codFrom, whiskerRight, whiskerLeft, box, Layer.cod, Layer.whiskerRight]
  have right : (whiskerLeft (S.dom f) (box g)).cod
      = (whiskerRight (box f) (S.cod g)).dom := by
    simp [cod, codFrom, whiskerRight, whiskerLeft, box, Layer.cod, Layer.whiskerLeft]
  constructor
  · intro h
    have layers := congrArg Diagram.layers h
    rw [layers_comp left, layers_comp right] at layers
    simp [whiskerRight, whiskerLeft, box, Layer.whiskerRight,
      Layer.whiskerLeft] at layers
    obtain ⟨⟨hdom, hfg, _⟩, hcod, _, _⟩ := layers
    exact ⟨hfg, hdom, hcod⟩
  · rintro ⟨rfl, hdom, hcod⟩
    ext
    · rw [dom_comp left, dom_comp right]; simp [whiskerRight, whiskerLeft, box, hdom]
    · rw [layers_comp left, layers_comp right]
      simp [whiskerRight, whiskerLeft, box, Layer.whiskerRight,
        Layer.whiskerLeft, hdom, hcod]

end Diagram

end Premonoidal
