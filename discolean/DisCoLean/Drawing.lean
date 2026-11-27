import DisCoLean.Premonoidal.Diagram

/-!
# The layered drawing of a diagram, a plane graph

The drawing with no optimisation: each layer of a diagram is a horizontal
strip, wire `k` of each row at `x = 2 * k + 1`, and the box of a layer
`(left, box, right)` covers `[2 * |left|, 2 * (|left| + |dom|)]` on top and
`[2 * |left|, 2 * (|left| + |cod|)]` at the bottom, so that a box with no
wires still sits between two wires. A strip is then given by four numbers,
the wires left of the box, its domain, its codomain and the wires right of
it, and its edges are straight segments, the wires passing the box, from
their `x` on top to their `x` at the bottom.

`Strip.plane`: the segments of a strip are ordered left to right at both
ends, so no two cross and the order of the wires is kept, and each is left of
the box at both ends or right of it at both ends, so none meets the box, the
trapezoid between its two sides: a segment and a side of the box move by the
same convex combination, as in `Isotopy.stays_apart`.

`Diagram.drawing` is a monoidal functor from Diagram(S) to these drawings:
composition is stacking strips (`drawing_comp`), whiskering on the left or
right adds wires on that side of every box (`drawing_whiskerLeft`,
`drawing_whiskerRight`), and the strips meet, the bottom of each having as
many wires as the top of the next (`Meets`).
-/

namespace Premonoidal

/-- A strip of a layered drawing: the wires left of its box, the domain and
codomain of the box, the wires right of it. -/
structure Strip where
  left : Nat
  dom : Nat
  cod : Nat
  right : Nat
deriving DecidableEq

namespace Strip

/-- The wires on top. -/
def top (s : Strip) : Nat := s.left + s.dom + s.right

/-- The wires at the bottom. -/
def bot (s : Strip) : Nat := s.left + s.cod + s.right

/-- The wires passing the box, each from its `x` on top to its `x` at the
bottom. -/
def segments (s : Strip) : List (Nat × Nat) :=
  (List.range s.left).map (fun k => (2 * k + 1, 2 * k + 1))
  ++ (List.range s.right).map (fun k =>
    (2 * (s.left + s.dom + k) + 1, 2 * (s.left + s.cod + k) + 1))

/-- The ports of the box on top. -/
def domPorts (s : Strip) : List Nat :=
  (List.range s.dom).map (fun i => 2 * (s.left + i) + 1)

/-- The ports of the box at the bottom. -/
def codPorts (s : Strip) : List Nat :=
  (List.range s.cod).map (fun i => 2 * (s.left + i) + 1)

/-- Two segments do not cross: the one is left of the other at both ends. -/
def Before (a b : Nat × Nat) : Prop := a.1 < b.1 ∧ a.2 < b.2

/-- A segment misses the box: left of it at both ends or right of it at both
ends. -/
def Misses (s : Strip) (a : Nat × Nat) : Prop :=
  a.1 < 2 * s.left ∧ a.2 < 2 * s.left
  ∨ 2 * (s.left + s.dom) < a.1 ∧ 2 * (s.left + s.cod) < a.2

/-- A strip is plane: its segments do not cross, in the order of the wires,
none meets the box, and the box has its ports on its sides. -/
structure Plane (s : Strip) : Prop where
  ordered : s.segments.Pairwise Before
  misses : ∀ a ∈ s.segments, s.Misses a
  domPorts : ∀ p ∈ s.domPorts, 2 * s.left ≤ p ∧ p ≤ 2 * (s.left + s.dom)
  codPorts : ∀ p ∈ s.codPorts, 2 * s.left ≤ p ∧ p ≤ 2 * (s.left + s.cod)

theorem plane (s : Strip) : s.Plane where
  ordered := by
    simp only [segments, List.pairwise_append, List.pairwise_map]
    refine ⟨(List.pairwise_lt_range).imp ?_, (List.pairwise_lt_range).imp ?_, ?_⟩
    · intro a b h; exact ⟨by omega, by omega⟩
    · intro a b h; exact ⟨by omega, by omega⟩
    · simp only [List.mem_map, List.mem_range]
      rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩
      exact ⟨by omega, by omega⟩
  misses := by
    simp only [segments, List.mem_append, List.mem_map, List.mem_range]
    rintro _ (⟨k, hk, rfl⟩ | ⟨k, hk, rfl⟩)
    · exact .inl ⟨by omega, by omega⟩
    · exact .inr ⟨by omega, by omega⟩
  domPorts := by
    simp only [domPorts, List.mem_map, List.mem_range]
    rintro _ ⟨i, hi, rfl⟩; omega
  codPorts := by
    simp only [codPorts, List.mem_map, List.mem_range]
    rintro _ ⟨i, hi, rfl⟩; omega

/-- Whiskering on the left adds `n` wires left of the box. -/
def shiftLeft (n : Nat) (s : Strip) : Strip := { s with left := n + s.left }

/-- Whiskering on the right adds `n` wires right of the box. -/
def shiftRight (s : Strip) (n : Nat) : Strip := { s with right := s.right + n }

end Strip

/-- The strips meet: from `n` wires on top, each strip starts where the one
above it ends. -/
def Meets : Nat → List Strip → Prop
  | _, [] => True
  | n, s :: ss => s.top = n ∧ Meets s.bot ss

variable {S : Signature}

/-- The strip of a layer. -/
def Layer.strip (l : Layer S) : Strip :=
  ⟨l.left.length, (S.dom l.box).length, (S.cod l.box).length, l.right.length⟩

theorem Layer.top_strip (l : Layer S) : l.strip.top = l.dom.length := by
  simp [strip, Strip.top, dom]; omega

theorem Layer.bot_strip (l : Layer S) : l.strip.bot = l.cod.length := by
  simp [strip, Strip.bot, cod]; omega

namespace Diagram

/-- The layered drawing of a diagram, its strips from top to bottom. -/
def drawing (f : Diagram S) : List Strip := f.layers.map Layer.strip

/-- Every strip of a drawing is plane. -/
theorem drawing_plane (f : Diagram S) : ∀ s ∈ f.drawing, s.Plane :=
  fun s _ => s.plane

theorem meets_of_composable {X : List S.Ob} {ls : List (Layer S)}
    (h : Composable X ls) : Meets X.length (ls.map Layer.strip) := by
  induction ls generalizing X with
  | nil => trivial
  | cons l ls ih =>
    exact ⟨by rw [Layer.top_strip, h.1], by rw [Layer.bot_strip]; exact ih h.2⟩

/-- The strips of a drawing meet, from as many wires as the domain. -/
theorem drawing_meets (f : Diagram S) : Meets f.dom.length f.drawing :=
  meets_of_composable f.composable

/-- A box is drawn as one strip with no wire passing it. -/
theorem drawing_box (b : S.Box) :
    (box b).drawing = [⟨0, (S.dom b).length, (S.cod b).length, 0⟩] := rfl

/-- The identity is drawn as no strip: its wires are the boundary. -/
theorem drawing_id (X : List S.Ob) : (id X).drawing = [] := rfl

/-- Composition is stacking. -/
theorem drawing_comp [DecidableEq S.Ob] {f g : Diagram S} (h : f.cod = g.dom) :
    (comp f g).drawing = f.drawing ++ g.drawing := by
  simp [drawing, layers_comp h]

/-- Whiskering on the left adds wires left of every box. -/
theorem drawing_whiskerLeft (Z : List S.Ob) (f : Diagram S) :
    (whiskerLeft Z f).drawing = f.drawing.map (Strip.shiftLeft Z.length) := by
  simp [drawing, whiskerLeft, Layer.whiskerLeft, Layer.strip, Strip.shiftLeft,
    Function.comp_def]

/-- Whiskering on the right adds wires right of every box. -/
theorem drawing_whiskerRight (f : Diagram S) (Z : List S.Ob) :
    (whiskerRight f Z).drawing = f.drawing.map (Strip.shiftRight · Z.length) := by
  simp [drawing, whiskerRight, Layer.whiskerRight, Layer.strip, Strip.shiftRight,
    Function.comp_def]

end Diagram

end Premonoidal
