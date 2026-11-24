import DisCoLean.Premonoidal.Diagram
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum.Basic
import Mathlib.Tactic.Linarith

/-!
# The `make_space` layout of `to_drawing` is plane

`Diagram.to_drawing` is a functor into drawings: a box is drawn by
`Drawing.from_box`, `@` by `Drawing.tensor`, which puts the second drawing on
the right of the first, and `>>` by `Drawing.then`, which puts the first on top
of the second and calls `make_space` until the outputs of the one are above
the inputs of the other. `make_space(s, x0)` shifts by `s` every node of one of
the two halves with `x ≥ x0`: in the model, `Layout.shift x0 s` on one of the
two layouts, before they are stacked.

The model keeps what decides whether a drawing is plane: every wire is a
vertical segment, as in every drawing `to_drawing` makes, and every box the
rectangle around its box node and its ports, a quarter wider on each side.
It leaves out the recentering of `reposition_box_cod` and `align_box_cod`
and the labels, which do not change the order of anything.

* `shift_expands`: the map of `make_space` never brings two points closer, so
  `shift_plane`: it keeps a layout plane.
* `then_plane` and `tensor_plane`: stacking two plane layouts, or putting
  them side by side, is plane, since the one lies above, or left of, the other.
* `align_aligned`: the loop of `Drawing.then` ends with each output of the top
  layout above the input of the bottom one with the same index.
* `layout_plane`: the layout of every diagram of Diagram(S) is plane.
-/

namespace MakeSpace

/-- A wire, the vertical segment at `x` from `lo` up to `hi`. -/
structure Wire where
  x : ℚ
  lo : ℚ
  hi : ℚ

/-- A box: the `x` of its box node and of its ports, the `y` of its box node
and half its height. -/
structure Rect where
  xs : List ℚ
  y : ℚ
  half : ℚ

/-- Two wires meet at most at an end: at different `x`, or one above the
other. -/
def WireApart (w v : Wire) : Prop :=
  w.x < v.x ∨ v.x < w.x ∨ w.hi ≤ v.lo ∨ v.hi ≤ w.lo

/-- `b` is left of `c`, their sides at most meeting. -/
def RectLeft (b c : Rect) : Prop := ∀ a ∈ b.xs, ∀ a' ∈ c.xs, a + 1 / 2 ≤ a'

/-- `b` is below `c`: the y axis points up, as in `to_drawing`. -/
def RectBelow (b c : Rect) : Prop := b.y + b.half ≤ c.y - c.half

/-- Two boxes overlap at most on a side. -/
def RectApart (b c : Rect) : Prop :=
  RectLeft b c ∨ RectLeft c b ∨ RectBelow b c ∨ RectBelow c b

/-- A wire misses a box: strictly left or right of it, or above or below it,
ending at most on its top or bottom side, where its ports are. -/
def WireRect (w : Wire) (b : Rect) : Prop :=
  (∀ a ∈ b.xs, w.x + 1 / 4 < a) ∨ (∀ a ∈ b.xs, a + 1 / 4 < w.x)
  ∨ w.hi ≤ b.y - b.half ∨ b.y + b.half ≤ w.lo

/-- A layout: its wires and boxes, the `x` of the wires on its top boundary
(`dom`) and bottom boundary (`cod`), its width and height. -/
structure Layout where
  wires : List Wire
  rects : List Rect
  dom : List ℚ
  cod : List ℚ
  width : ℚ
  height : ℚ

/-- A plane layout: no two wires meet but at an end, no two boxes overlap and
no wire passes through a box. -/
structure Plane (L : Layout) : Prop where
  wires : L.wires.Pairwise WireApart
  rects : L.rects.Pairwise RectApart
  wireRect : ∀ w ∈ L.wires, ∀ b ∈ L.rects, WireRect w b

/-- A layout lies in its frame, a quarter away from its left and right
sides, and its boundary wires are in order from left to right. -/
structure Bounded (L : Layout) : Prop where
  wire_x : ∀ w ∈ L.wires, 1 / 4 ≤ w.x ∧ w.x + 1 / 4 ≤ L.width
  wire_y : ∀ w ∈ L.wires, 0 ≤ w.lo ∧ w.lo ≤ w.hi ∧ w.hi ≤ L.height
  rect_x : ∀ b ∈ L.rects, b.xs ≠ [] ∧ ∀ a ∈ b.xs, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ L.width
  rect_y : ∀ b ∈ L.rects, 0 ≤ b.half ∧ 0 ≤ b.y - b.half ∧ b.y + b.half ≤ L.height
  dom : ∀ a ∈ L.dom, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ L.width
  cod : ∀ a ∈ L.cod, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ L.width
  dom_sorted : L.dom.Pairwise (· < ·)
  cod_sorted : L.cod.Pairwise (· < ·)
  height : 0 ≤ L.height
  width : 0 ≤ L.width

/-! ## `make_space` -/

theorem ite_t {α : Type} {c : Prop} [Decidable c] (h : c) (a b : α) :
    (if c then a else b) = a := by simp [h]

theorem ite_f {α : Type} {c : Prop} [Decidable c] (h : ¬ c) (a b : α) :
    (if c then a else b) = b := by simp [h]

/-- The map of `make_space(s, x0)` on the `x` of a node. -/
def shiftX (x0 s x : ℚ) : ℚ := if x0 ≤ x then x + s else x

/-- It never brings two points closer. -/
theorem shift_expands {x0 s a b : ℚ} (hs : 0 ≤ s) (h : a ≤ b) :
    b - a ≤ shiftX x0 s b - shiftX x0 s a := by
  unfold shiftX; split_ifs <;> linarith

theorem shift_le {x0 s a b c : ℚ} (hs : 0 ≤ s) (hc : 0 ≤ c) (h : a + c ≤ b) :
    shiftX x0 s a + c ≤ shiftX x0 s b := by
  have := shift_expands (x0 := x0) hs (show a ≤ b by linarith); linarith

theorem shift_lt {x0 s a b c : ℚ} (hs : 0 ≤ s) (hc : 0 ≤ c) (h : a + c < b) :
    shiftX x0 s a + c < shiftX x0 s b := by
  have := shift_expands (x0 := x0) hs (show a ≤ b by linarith); linarith

theorem shift_strict {x0 s a b : ℚ} (hs : 0 ≤ s) (h : a < b) :
    shiftX x0 s a < shiftX x0 s b := by
  simpa using shift_lt (x0 := x0) (c := 0) hs le_rfl (by simpa using h)

theorem le_shift {x0 s a : ℚ} (hs : 0 ≤ s) : a ≤ shiftX x0 s a := by
  unfold shiftX; split_ifs <;> linarith

theorem shift_le_add {x0 s a : ℚ} (hs : 0 ≤ s) : shiftX x0 s a ≤ a + s := by
  unfold shiftX; split_ifs <;> linarith

def Wire.shift (x0 s : ℚ) (w : Wire) : Wire := { w with x := shiftX x0 s w.x }

def Rect.shift (x0 s : ℚ) (b : Rect) : Rect := { b with xs := b.xs.map (shiftX x0 s) }

/-- `make_space(s, x0)` on a whole layout, one of the two halves of `then`. -/
def Layout.shift (x0 s : ℚ) (L : Layout) : Layout where
  wires := L.wires.map (Wire.shift x0 s)
  rects := L.rects.map (Rect.shift x0 s)
  dom := L.dom.map (shiftX x0 s)
  cod := L.cod.map (shiftX x0 s)
  width := L.width + s
  height := L.height

theorem shift_sorted {x0 s : ℚ} (hs : 0 ≤ s) {l : List ℚ} (h : l.Pairwise (· < ·)) :
    (l.map (shiftX x0 s)).Pairwise (· < ·) :=
  List.pairwise_map.2 (h.imp (shift_strict hs))

/-- `make_space` keeps a layout plane. -/
theorem shift_plane {x0 s : ℚ} (hs : 0 ≤ s) {L : Layout} (h : Plane L) :
    Plane (L.shift x0 s) where
  wires := List.pairwise_map.2 <| h.wires.imp fun {w v} h => by
    rcases h with h | h | h | h
    · exact .inl (shift_strict hs h)
    · exact .inr (.inl (shift_strict hs h))
    · exact .inr (.inr (.inl h))
    · exact .inr (.inr (.inr h))
  rects := List.pairwise_map.2 <| h.rects.imp fun {b c} h => by
    rcases h with h | h | h | h
    · refine .inl ?_
      simp only [RectLeft, Rect.shift, List.mem_map]
      rintro _ ⟨a, ha, rfl⟩ _ ⟨a', ha', rfl⟩
      exact shift_le hs (by norm_num) (h a ha a' ha')
    · refine .inr (.inl ?_)
      simp only [RectLeft, Rect.shift, List.mem_map]
      rintro _ ⟨a, ha, rfl⟩ _ ⟨a', ha', rfl⟩
      exact shift_le hs (by norm_num) (h a ha a' ha')
    · exact .inr (.inr (.inl h))
    · exact .inr (.inr (.inr h))
  wireRect := by
    simp only [Layout.shift, List.mem_map]
    rintro _ ⟨w, hw, rfl⟩ _ ⟨b, hb, rfl⟩
    rcases h.wireRect w hw b hb with h | h | h | h
    · refine .inl ?_
      simp only [Rect.shift, Wire.shift, List.mem_map]
      rintro _ ⟨a, ha, rfl⟩
      exact shift_lt hs (by norm_num) (h a ha)
    · refine .inr (.inl ?_)
      simp only [Rect.shift, Wire.shift, List.mem_map]
      rintro _ ⟨a, ha, rfl⟩
      exact shift_lt hs (by norm_num) (h a ha)
    · exact .inr (.inr (.inl h))
    · exact .inr (.inr (.inr h))

theorem shift_bounded {x0 s : ℚ} (hs : 0 ≤ s) {L : Layout} (h : Bounded L) :
    Bounded (L.shift x0 s) := by
  have bound : ∀ a, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ L.width →
      1 / 4 ≤ shiftX x0 s a ∧ shiftX x0 s a + 1 / 4 ≤ L.width + s := fun a ha =>
    ⟨le_trans ha.1 (le_shift hs), by linarith [shift_le_add (x0 := x0) (a := a) hs]⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, shift_sorted hs h.dom_sorted, shift_sorted hs h.cod_sorted,
    h.height, by simp only [Layout.shift]; linarith [h.width]⟩
  · simp only [Layout.shift, List.mem_map]
    rintro _ ⟨w, hw, rfl⟩; exact bound _ (h.wire_x w hw)
  · simp only [Layout.shift, List.mem_map]
    rintro _ ⟨w, hw, rfl⟩; exact h.wire_y w hw
  · simp only [Layout.shift, List.mem_map]
    rintro _ ⟨b, hb, rfl⟩
    refine ⟨by simpa [Rect.shift] using (h.rect_x b hb).1, ?_⟩
    simp only [Rect.shift, List.mem_map]
    rintro _ ⟨a, ha, rfl⟩; exact bound _ ((h.rect_x b hb).2 a ha)
  · simp only [Layout.shift, List.mem_map]
    rintro _ ⟨b, hb, rfl⟩; exact h.rect_y b hb
  · simp only [Layout.shift, List.mem_map]
    rintro _ ⟨a, ha, rfl⟩; exact bound _ (h.dom a ha)
  · simp only [Layout.shift, List.mem_map]
    rintro _ ⟨a, ha, rfl⟩; exact bound _ (h.cod a ha)

/-! ## Stacking and side by side -/

def Wire.up (d : ℚ) (w : Wire) : Wire := { w with lo := w.lo + d, hi := w.hi + d }
def Rect.up (d : ℚ) (b : Rect) : Rect := { b with y := b.y + d }
def Wire.right (d : ℚ) (w : Wire) : Wire := { w with x := w.x + d }
def Rect.right (d : ℚ) (b : Rect) : Rect := { b with xs := b.xs.map (· + d) }

/-- `L` on top of `M`, as `Drawing.then` stacks them once aligned. -/
def Layout.then (L M : Layout) : Layout where
  wires := L.wires.map (Wire.up M.height) ++ M.wires
  rects := L.rects.map (Rect.up M.height) ++ M.rects
  dom := L.dom
  cod := M.cod
  width := max L.width M.width
  height := L.height + M.height

/-- `L` then `M` on its right, `gap` apart, as `Drawing.tensor` puts them
once of one height. -/
def Layout.tensor (L M : Layout) (gap : ℚ) : Layout where
  wires := L.wires ++ M.wires.map (Wire.right (L.width + gap))
  rects := L.rects ++ M.rects.map (Rect.right (L.width + gap))
  dom := L.dom ++ M.dom.map (· + (L.width + gap))
  cod := L.cod ++ M.cod.map (· + (L.width + gap))
  width := L.width + gap + M.width
  height := L.height

/-- Stacking two plane layouts is plane: the one lies above the other. -/
theorem then_plane {L M : Layout} (hL : Plane L) (hM : Plane M)
    (bL : Bounded L) (bM : Bounded M) : Plane (L.then M) where
  wires := by
    refine List.pairwise_append.2 ⟨List.pairwise_map.2 (hL.wires.imp fun {w v} h => ?_),
      hM.wires, ?_⟩
    · rcases h with h | h | h | h
      · exact .inl h
      · exact .inr (.inl h)
      · exact .inr (.inr (.inl (by simp only [Wire.up]; linarith)))
      · exact .inr (.inr (.inr (by simp only [Wire.up]; linarith)))
    · simp only [List.mem_map]
      rintro _ ⟨w, hw, rfl⟩ v hv
      exact .inr (.inr (.inr (by
        simp only [Wire.up]; linarith [(bL.wire_y w hw).1, (bM.wire_y v hv).2.2])))
  rects := by
    refine List.pairwise_append.2 ⟨List.pairwise_map.2 (hL.rects.imp fun {b c} h => ?_),
      hM.rects, ?_⟩
    · rcases h with h | h | h | h
      · exact .inl h
      · exact .inr (.inl h)
      · exact .inr (.inr (.inl (by simp only [RectBelow, Rect.up] at h ⊢; linarith)))
      · exact .inr (.inr (.inr (by simp only [RectBelow, Rect.up] at h ⊢; linarith)))
    · simp only [List.mem_map]
      rintro _ ⟨b, hb, rfl⟩ c hc
      exact .inr (.inr (.inr (by
        simp only [RectBelow, Rect.up]
        linarith [(bL.rect_y b hb).2.1, (bM.rect_y c hc).2.2])))
  wireRect := by
    simp only [Layout.then, List.mem_append, List.mem_map]
    rintro w' (⟨w, hw, rfl⟩ | hw) b' (⟨b, hb, rfl⟩ | hb)
    · rcases hL.wireRect w hw b hb with h | h | h | h
      · exact .inl h
      · exact .inr (.inl h)
      · exact .inr (.inr (.inl (by simp only [Wire.up, Rect.up]; linarith)))
      · exact .inr (.inr (.inr (by simp only [Wire.up, Rect.up]; linarith)))
    · exact .inr (.inr (.inr (by
        simp only [Wire.up]; linarith [(bL.wire_y w hw).1, (bM.rect_y b' hb).2.2])))
    · exact .inr (.inr (.inl (by
        simp only [Rect.up]; linarith [(bL.rect_y b hb).2.1, (bM.wire_y w' hw).2.2])))
    · exact hM.wireRect w' hw b' hb

theorem then_bounded {L M : Layout} (bL : Bounded L) (bM : Bounded M) :
    Bounded (L.then M) := by
  have wL := le_max_left L.width M.width
  have wM := le_max_right L.width M.width
  have hL := bL.height
  have hM := bM.height
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, bL.dom_sorted, bM.cod_sorted, by
    simp only [Layout.then]; linarith, by simp only [Layout.then]; linarith [bL.width]⟩
  · simp only [Layout.then, List.mem_append, List.mem_map]
    rintro _ (⟨w, hw, rfl⟩ | hw)
    · exact ⟨(bL.wire_x w hw).1, by simp only [Wire.up]; linarith [(bL.wire_x w hw).2]⟩
    · exact ⟨(bM.wire_x _ hw).1, by linarith [(bM.wire_x _ hw).2]⟩
  · simp only [Layout.then, List.mem_append, List.mem_map]
    rintro _ (⟨w, hw, rfl⟩ | hw)
    · obtain ⟨h0, h1, h2⟩ := bL.wire_y w hw
      simp only [Wire.up]; exact ⟨by linarith, by linarith, by linarith⟩
    · obtain ⟨h0, h1, h2⟩ := bM.wire_y _ hw
      exact ⟨h0, h1, by linarith⟩
  · simp only [Layout.then, List.mem_append, List.mem_map]
    rintro _ (⟨b, hb, rfl⟩ | hb)
    · exact ⟨by simpa [Rect.up] using (bL.rect_x b hb).1, fun a ha => by
        have := (bL.rect_x b hb).2 a (by simpa [Rect.up] using ha)
        exact ⟨this.1, by linarith [this.2]⟩⟩
    · exact ⟨(bM.rect_x _ hb).1, fun a ha => by
        have := (bM.rect_x _ hb).2 a ha
        exact ⟨this.1, by linarith [this.2]⟩⟩
  · simp only [Layout.then, List.mem_append, List.mem_map]
    rintro _ (⟨b, hb, rfl⟩ | hb)
    · obtain ⟨h0, h1, h2⟩ := bL.rect_y b hb
      simp only [Rect.up]; exact ⟨h0, by linarith, by linarith⟩
    · obtain ⟨h0, h1, h2⟩ := bM.rect_y _ hb
      exact ⟨h0, h1, by linarith⟩
  · intro a ha; exact ⟨(bL.dom a ha).1, by simp only [Layout.then]; linarith [(bL.dom a ha).2]⟩
  · intro a ha; exact ⟨(bM.cod a ha).1, by simp only [Layout.then]; linarith [(bM.cod a ha).2]⟩

/-- Putting two plane layouts side by side is plane: the one lies left of
the other. -/
theorem tensor_plane {L M : Layout} {gap : ℚ} (hgap : 0 ≤ gap) (hL : Plane L)
    (hM : Plane M) (bL : Bounded L) (bM : Bounded M) : Plane (L.tensor M gap) where
  wires := by
    refine List.pairwise_append.2 ⟨hL.wires, List.pairwise_map.2 (hM.wires.imp fun {w v} h => ?_), ?_⟩
    · rcases h with h | h | h | h
      · exact .inl (by simp only [Wire.right]; linarith)
      · exact .inr (.inl (by simp only [Wire.right]; linarith))
      · exact .inr (.inr (.inl h))
      · exact .inr (.inr (.inr h))
    · simp only [List.mem_map]
      rintro w hw _ ⟨v, hv, rfl⟩
      exact .inl (by
        simp only [Wire.right]; linarith [(bL.wire_x w hw).2, (bM.wire_x v hv).1])
  rects := by
    refine List.pairwise_append.2 ⟨hL.rects, List.pairwise_map.2 (hM.rects.imp fun {b c} h => ?_), ?_⟩
    · rcases h with h | h | h | h
      · refine .inl ?_
        simp only [RectLeft, Rect.right, List.mem_map]
        rintro _ ⟨a, ha, rfl⟩ _ ⟨a', ha', rfl⟩; linarith [h a ha a' ha']
      · refine .inr (.inl ?_)
        simp only [RectLeft, Rect.right, List.mem_map]
        rintro _ ⟨a, ha, rfl⟩ _ ⟨a', ha', rfl⟩; linarith [h a ha a' ha']
      · exact .inr (.inr (.inl h))
      · exact .inr (.inr (.inr h))
    · simp only [List.mem_map]
      rintro b hb _ ⟨c, hc, rfl⟩
      refine .inl ?_
      simp only [RectLeft, Rect.right, List.mem_map]
      rintro a ha _ ⟨a', ha', rfl⟩
      linarith [((bL.rect_x b hb).2 a ha).2, ((bM.rect_x c hc).2 a' ha').1]
  wireRect := by
    simp only [Layout.tensor, List.mem_append, List.mem_map]
    rintro w' (hw | ⟨w, hw, rfl⟩) b' (hb | ⟨b, hb, rfl⟩)
    · exact hL.wireRect w' hw b' hb
    · refine .inl ?_
      simp only [Rect.right, List.mem_map]
      rintro _ ⟨a, ha, rfl⟩
      linarith [(bL.wire_x w' hw).2, ((bM.rect_x b hb).2 a ha).1]
    · refine .inr (.inl ?_)
      intro a ha
      simp only [Wire.right]
      linarith [((bL.rect_x b' hb).2 a ha).2, (bM.wire_x w hw).1]
    · rcases hM.wireRect w hw b hb with h | h | h | h
      · refine .inl ?_
        simp only [Rect.right, Wire.right, List.mem_map]
        rintro _ ⟨a, ha, rfl⟩; linarith [h a ha]
      · refine .inr (.inl ?_)
        simp only [Rect.right, Wire.right, List.mem_map]
        rintro _ ⟨a, ha, rfl⟩; linarith [h a ha]
      · exact .inr (.inr (.inl h))
      · exact .inr (.inr (.inr h))

theorem tensor_sorted {L : Layout} {gap : ℚ} (hgap : 0 ≤ gap) {l m : List ℚ}
    (hl : l.Pairwise (· < ·)) (hm : m.Pairwise (· < ·))
    (bl : ∀ a ∈ l, a + 1 / 4 ≤ L.width) (bm : ∀ a ∈ m, 1 / 4 ≤ a) :
    (l ++ m.map (· + (L.width + gap))).Pairwise (· < ·) := by
  refine List.pairwise_append.2 ⟨hl, List.pairwise_map.2 (hm.imp fun h => by linarith), ?_⟩
  simp only [List.mem_map]
  rintro a ha _ ⟨b, hb, rfl⟩; linarith [bl a ha, bm b hb]

theorem tensor_bounded {L M : Layout} {gap : ℚ} (hgap : 0 ≤ gap)
    (heq : L.height = M.height) (bL : Bounded L) (bM : Bounded M) :
    Bounded (L.tensor M gap) := by
  have w0 := bL.width
  have m0 := bM.width
  have left : ∀ a, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ L.width →
      1 / 4 ≤ a ∧ a + 1 / 4 ≤ L.width + gap + M.width :=
    fun a ha => ⟨ha.1, by linarith [ha.2]⟩
  have right : ∀ a, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ M.width →
      1 / 4 ≤ a + (L.width + gap) ∧ a + (L.width + gap) + 1 / 4 ≤ L.width + gap + M.width :=
    fun a ha => ⟨by linarith [ha.1], by linarith [ha.2]⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_,
    tensor_sorted hgap bL.dom_sorted bM.dom_sorted (fun a ha => (bL.dom a ha).2)
      (fun a ha => (bM.dom a ha).1),
    tensor_sorted hgap bL.cod_sorted bM.cod_sorted (fun a ha => (bL.cod a ha).2)
      (fun a ha => (bM.cod a ha).1),
    bL.height, by simp only [Layout.tensor]; linarith⟩
  · simp only [Layout.tensor, List.mem_append, List.mem_map]
    rintro _ (hw | ⟨w, hw, rfl⟩)
    · exact left _ (bL.wire_x _ hw)
    · exact right _ (bM.wire_x w hw)
  · simp only [Layout.tensor, List.mem_append, List.mem_map]
    rintro _ (hw | ⟨w, hw, rfl⟩)
    · exact bL.wire_y _ hw
    · exact heq ▸ bM.wire_y w hw
  · simp only [Layout.tensor, List.mem_append, List.mem_map]
    rintro _ (hb | ⟨b, hb, rfl⟩)
    · exact ⟨(bL.rect_x _ hb).1, fun a ha => left a ((bL.rect_x _ hb).2 a ha)⟩
    · refine ⟨by simpa [Rect.right] using (bM.rect_x b hb).1, ?_⟩
      simp only [Rect.right, List.mem_map]
      rintro _ ⟨a, ha, rfl⟩; exact right a ((bM.rect_x b hb).2 a ha)
  · simp only [Layout.tensor, List.mem_append, List.mem_map]
    rintro _ (hb | ⟨b, hb, rfl⟩)
    · exact bL.rect_y _ hb
    · exact heq ▸ bM.rect_y b hb
  · simp only [Layout.tensor, List.mem_append, List.mem_map]
    rintro _ (ha | ⟨a, ha, rfl⟩)
    · exact left _ (bL.dom _ ha)
    · exact right a (bM.dom a ha)
  · simp only [Layout.tensor, List.mem_append, List.mem_map]
    rintro _ (ha | ⟨a, ha, rfl⟩)
    · exact left _ (bL.cod _ ha)
    · exact right a (bM.cod a ha)

/-! ## The loop of `Drawing.then` -/

/-- One turn of the loop of `Drawing.then`, on wire `i`: the half whose end
of that wire is left of the other's is shifted by `make_space` from that end,
from `0` for the first wire, until the two ends are one above the other. -/
def alignAt (i : ℕ) (p : Layout × Layout) : Layout × Layout :=
  let top := p.1.cod.getD i 0
  let bot := p.2.dom.getD i 0
  if bot < top then (p.1, p.2.shift (if i = 0 then 0 else bot) (top - bot))
  else if top < bot then (p.1.shift (if i = 0 then 0 else top) (bot - top), p.2)
  else p

/-- The loop of `Drawing.then`, over the inputs of `M`. -/
def align (L M : Layout) : Layout × Layout :=
  (List.range M.dom.length).foldl (fun p i => alignAt i p) (L, M)

/-- `L >> M`: aligned, then stacked. -/
def compose (L M : Layout) : Layout := (align L M).1.then (align L M).2

/-- A layout fit to compose: plane and in its frame. -/
def Fit (L : Layout) : Prop := Plane L ∧ Bounded L

theorem alignAt_cases (i : ℕ) (p : Layout × Layout) : alignAt i p = p ∨ ∃ x0 s, 0 ≤ s ∧
    (alignAt i p = (p.1, p.2.shift x0 s) ∨ alignAt i p = (p.1.shift x0 s, p.2)) := by
  unfold alignAt; dsimp only
  by_cases h : p.2.dom.getD i 0 < p.1.cod.getD i 0
  · rw [ite_t h]; exact .inr ⟨_, _, by linarith, .inl rfl⟩
  by_cases h' : p.1.cod.getD i 0 < p.2.dom.getD i 0
  · rw [ite_f h, ite_t h']; exact .inr ⟨_, _, by linarith, .inr rfl⟩
  · rw [ite_f h, ite_f h']; exact .inl rfl

theorem alignAt_fit (i : ℕ) {p : Layout × Layout} (h1 : Fit p.1) (h2 : Fit p.2) :
    Fit (alignAt i p).1 ∧ Fit (alignAt i p).2 := by
  rcases alignAt_cases i p with e | ⟨x0, s, hs, e | e⟩ <;> rw [e]
  · exact ⟨h1, h2⟩
  · exact ⟨h1, shift_plane hs h2.1, shift_bounded hs h2.2⟩
  · exact ⟨⟨shift_plane hs h1.1, shift_bounded hs h1.2⟩, h2⟩

theorem foldl_alignAt_fit (is : List ℕ) {p : Layout × Layout} (h1 : Fit p.1) (h2 : Fit p.2) :
    Fit (is.foldl (fun p i => alignAt i p) p).1 ∧ Fit (is.foldl (fun p i => alignAt i p) p).2 := by
  induction is generalizing p with
  | nil => exact ⟨h1, h2⟩
  | cons i is ih => exact ih (alignAt_fit i h1 h2).1 (alignAt_fit i h1 h2).2

/-- Composing two plane layouts in their frames is plane and in its frame. -/
theorem compose_fit {L M : Layout} (hL : Fit L) (hM : Fit M) : Fit (compose L M) := by
  obtain ⟨h1, h2⟩ := foldl_alignAt_fit (List.range M.dom.length) (p := (L, M)) hL hM
  exact ⟨then_plane h1.1 h2.1 h1.2 h2.2, then_bounded h1.2 h2.2⟩

theorem alignAt_length (i : ℕ) (p : Layout × Layout) :
    (alignAt i p).1.cod.length = p.1.cod.length ∧ (alignAt i p).2.dom.length = p.2.dom.length := by
  rcases alignAt_cases i p with e | ⟨x0, s, hs, e | e⟩ <;> rw [e] <;> simp [Layout.shift]

/-- The shift of a turn starts at most at the end it moves, and exactly at it
but on the first wire. -/
theorem shiftX_fixes {k : ℕ} {x0 bot a s : ℚ} (hx0 : x0 = if k = 0 then 0 else bot)
    (hbot : 0 ≤ bot) : shiftX x0 s bot = bot + s ∧ (0 < k → a < bot → shiftX x0 s a = a) := by
  constructor
  · unfold shiftX; rw [ite_t (by subst hx0; split_ifs <;> linarith)]
  · intro hk ha
    unfold shiftX; rw [ite_f (by subst hx0; rw [ite_f (by omega)]; linarith)]

theorem alignAt_step {p : Layout × Layout} {k : ℕ}
    (hkc : k < p.1.cod.length) (hkd : k < p.2.dom.length)
    (b1 : Bounded p.1) (b2 : Bounded p.2)
    (h : ∀ j < k, p.1.cod[j]? = p.2.dom[j]?) :
    ∀ j < k + 1, (alignAt k p).1.cod[j]? = (alignAt k p).2.dom[j]? := by
  have etop : p.1.cod.getD k 0 = p.1.cod[k] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkc]; rfl
  have ebot : p.2.dom.getD k 0 = p.2.dom[k] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkd]; rfl
  have ptop := (b1.cod _ (List.getElem_mem hkc)).1
  have pbot := (b2.dom _ (List.getElem_mem hkd)).1
  intro j hj
  unfold alignAt; dsimp only; rw [etop, ebot]
  by_cases h1 : p.2.dom[k] < p.1.cod[k]
  · rw [ite_t h1]
    have fix := shiftX_fixes (k := k) (a := 0) (s := p.1.cod[k] - p.2.dom[k]) rfl (by linarith)
    simp only [Layout.shift, List.getElem?_map]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · have hjd : j < p.2.dom.length := by omega
      rw [h j hj, List.getElem?_eq_getElem hjd]
      have lt := List.pairwise_iff_getElem.1 b2.dom_sorted j k hjd hkd hj
      have := (shiftX_fixes (k := k) (a := p.2.dom[j]) (s := p.1.cod[k] - p.2.dom[k]) rfl
        (by linarith)).2 (by omega) lt
      simp only [Option.map_some, this]
    · rw [List.getElem?_eq_getElem hkc, List.getElem?_eq_getElem hkd]
      simp only [Option.map_some, fix.1, Option.some.injEq]; ring
  by_cases h2 : p.1.cod[k] < p.2.dom[k]
  · rw [ite_f h1, ite_t h2]
    have fix := shiftX_fixes (k := k) (bot := p.1.cod[k]) (a := 0) (s := p.2.dom[k] - p.1.cod[k])
      rfl (by linarith)
    simp only [Layout.shift, List.getElem?_map]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · have hjc : j < p.1.cod.length := by omega
      rw [← h j hj, List.getElem?_eq_getElem hjc]
      have lt := List.pairwise_iff_getElem.1 b1.cod_sorted j k hjc hkc hj
      have := (shiftX_fixes (k := k) (bot := p.1.cod[k]) (a := p.1.cod[j])
        (s := p.2.dom[k] - p.1.cod[k]) rfl
        (by linarith)).2 (by omega) lt
      simp only [Option.map_some, this]
    · rw [List.getElem?_eq_getElem hkc, List.getElem?_eq_getElem hkd]
      simp only [Option.map_some, fix.1, Option.some.injEq]; ring
  · rw [ite_f h1, ite_f h2]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · exact h j hj
    · rw [List.getElem?_eq_getElem hkc, List.getElem?_eq_getElem hkd]
      congr 1; linarith [not_lt.1 h1, not_lt.1 h2]

theorem align_inv {L M : Layout} (hn : L.cod.length = M.dom.length)
    (bL : Bounded L) (bM : Bounded M) (hL : Plane L) (hM : Plane M) :
    ∀ k ≤ M.dom.length,
      let p := (List.range k).foldl (fun p i => alignAt i p) (L, M)
      p.1.cod.length = M.dom.length ∧ p.2.dom.length = M.dom.length
      ∧ ∀ j < k, p.1.cod[j]? = p.2.dom[j]? := by
  intro k hk
  induction k with
  | zero => exact ⟨hn, rfl, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
  | succ k ih =>
    obtain ⟨hc, hd, h⟩ := ih (by omega)
    have fit := foldl_alignAt_fit (List.range k) (p := (L, M)) ⟨hL, bL⟩ ⟨hM, bM⟩
    simp only [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    have len := alignAt_length k ((List.range k).foldl (fun p i => alignAt i p) (L, M))
    exact ⟨len.1.trans hc, len.2.trans hd,
      alignAt_step (by omega) (by omega) fit.1.2 fit.2.2 h⟩

/-- The loop of `Drawing.then` ends with every output of the top layout
above the input of the bottom one with the same index. -/
theorem align_aligned {L M : Layout} (hn : L.cod.length = M.dom.length)
    (hL : Fit L) (hM : Fit M) : (align L M).1.cod = (align L M).2.dom := by
  obtain ⟨hc, hd, h⟩ := align_inv hn hL.2 hM.2 hL.1 hM.1 M.dom.length le_rfl
  apply List.ext_getElem?
  intro j
  by_cases hj : j < M.dom.length
  · exact h j hj
  · unfold align
    rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by omega)]

/-! ## Boxes, identities and layers -/

/-- The `x` of `c` wires centred on a box `w` wide, one apart. -/
def ports (w : ℚ) (c : ℕ) : List ℚ := (List.range c).map fun k : ℕ => (w - c) / 2 + (k : ℚ) + 1 / 2

/-- `Drawing.from_box` of a box with `n` inputs and `m` outputs: as wide as
its widest side and at least `1`, its box node in the middle, its ports a
quarter above and below it. -/
def boxLayout (n m : ℕ) : Layout :=
  let w : ℚ := max (max (n : ℚ) (m : ℚ)) 1
  { wires := (ports w n).map (fun x => ⟨x, 3 / 4, 1⟩) ++ (ports w m).map (fun x => ⟨x, 0, 1 / 4⟩)
    rects := [⟨ports w n ++ ports w m ++ [w / 2], 1 / 2, 1 / 4⟩]
    dom := ports w n
    cod := ports w m
    width := w
    height := 1 }

/-- `Drawing.id` of `c` wires, `h` high. -/
def idLayout (c : ℕ) (h : ℚ) : Layout :=
  { wires := (ports c c).map (fun x => ⟨x, 0, h⟩)
    rects := []
    dom := ports c c
    cod := ports c c
    width := c
    height := h }

theorem ports_sorted (w : ℚ) (c : ℕ) : (ports w c).Pairwise (· < ·) := by
  unfold ports
  refine List.pairwise_map.2 (List.pairwise_lt_range.imp fun {a b} h => ?_)
  have : (a : ℚ) < b := by exact_mod_cast h
  linarith

theorem ports_bounds {w : ℚ} {c : ℕ} (hc : (c : ℚ) ≤ w) :
    ∀ a ∈ ports w c, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ w := by
  simp only [ports, List.mem_map, List.mem_range]
  rintro _ ⟨k, hk, rfl⟩
  have : (k : ℚ) + 1 ≤ c := by exact_mod_cast hk
  have : (0 : ℚ) ≤ k := Nat.cast_nonneg k
  constructor <;> linarith

theorem ports_pairwise_wires (w lo hi : ℚ) (c : ℕ) :
    ((ports w c).map (fun x => (⟨x, lo, hi⟩ : Wire))).Pairwise WireApart :=
  List.pairwise_map.2 ((ports_sorted w c).imp fun h => .inl h)

theorem boxLayout_fit (n m : ℕ) : Fit (boxLayout n m) := by
  have hn : (n : ℚ) ≤ max (max (n : ℚ) (m : ℚ)) 1 :=
    le_trans (le_max_left _ _) (le_max_left _ _)
  have hm : (m : ℚ) ≤ max (max (n : ℚ) (m : ℚ)) 1 :=
    le_trans (le_max_right _ _) (le_max_left _ _)
  have h1 : (1 : ℚ) ≤ max (max (n : ℚ) (m : ℚ)) 1 := le_max_right _ _
  refine ⟨⟨?_, List.pairwise_singleton _ _, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ports_sorted _ _,
    ports_sorted _ _, by simp [boxLayout], by simp only [boxLayout]; linarith⟩⟩
  · refine List.pairwise_append.2 ⟨ports_pairwise_wires _ _ _ _, ports_pairwise_wires _ _ _ _, ?_⟩
    simp only [List.mem_map]
    rintro _ ⟨x, _, rfl⟩ _ ⟨y, _, rfl⟩
    exact .inr (.inr (.inr (by norm_num)))
  · simp only [boxLayout, List.mem_append, List.mem_map, List.mem_singleton]
    rintro _ (⟨x, _, rfl⟩ | ⟨x, _, rfl⟩) _ rfl
    · exact .inr (.inr (.inr (by norm_num)))
    · exact .inr (.inr (.inl (by norm_num)))
  · simp only [boxLayout, List.mem_append, List.mem_map]
    rintro _ (⟨x, hx, rfl⟩ | ⟨x, hx, rfl⟩)
    · exact ports_bounds hn x hx
    · exact ports_bounds hm x hx
  · simp only [boxLayout, List.mem_append, List.mem_map]
    rintro _ (⟨x, hx, rfl⟩ | ⟨x, hx, rfl⟩) <;> norm_num
  · simp only [boxLayout, List.mem_singleton]
    rintro _ rfl
    refine ⟨by simp, ?_⟩
    simp only [List.mem_append, List.mem_singleton]
    rintro a ((ha | ha) | rfl)
    · exact ports_bounds hn a ha
    · exact ports_bounds hm a ha
    · constructor <;> linarith
  · simp only [boxLayout, List.mem_singleton]
    rintro _ rfl; norm_num
  · exact ports_bounds hn
  · exact ports_bounds hm

theorem idLayout_fit (c : ℕ) {h : ℚ} (hh : 0 ≤ h) : Fit (idLayout c h) := by
  have hc : (c : ℚ) ≤ c := le_rfl
  refine ⟨⟨ports_pairwise_wires _ _ _ _, List.Pairwise.nil, by simp [idLayout]⟩,
    ⟨?_, ?_, by simp [idLayout], by simp [idLayout], ports_bounds hc, ports_bounds hc,
      ports_sorted _ _, ports_sorted _ _, hh, by simp [idLayout]⟩⟩
  · simp only [idLayout, List.mem_map]
    rintro _ ⟨x, hx, rfl⟩; exact ports_bounds hc x hx
  · simp only [idLayout, List.mem_map]
    rintro _ ⟨x, hx, rfl⟩; exact ⟨le_rfl, hh, le_rfl⟩

end MakeSpace

namespace Premonoidal

open MakeSpace

variable {S : Signature}

/-- The layout of a layer: the wires on its left, its box, the wires on its
right, side by side. -/
def Layer.layout (l : Layer S) : Layout :=
  (idLayout l.left.length 1).tensor
    ((boxLayout (S.dom l.box).length (S.cod l.box).length).tensor
      (idLayout l.right.length 1) 0) 0

theorem Layer.layout_fit (l : Layer S) : Fit l.layout := by
  have hb := boxLayout_fit (S.dom l.box).length (S.cod l.box).length
  have hl := idLayout_fit l.left.length (h := 1) (by norm_num)
  have hr := idLayout_fit l.right.length (h := 1) (by norm_num)
  have hbr : Fit ((boxLayout (S.dom l.box).length (S.cod l.box).length).tensor
      (idLayout l.right.length 1) 0) :=
    ⟨tensor_plane le_rfl hb.1 hr.1 hb.2 hr.2, tensor_bounded le_rfl rfl hb.2 hr.2⟩
  exact ⟨tensor_plane le_rfl hl.1 hbr.1 hl.2 hbr.2, tensor_bounded le_rfl rfl hl.2 hbr.2⟩

namespace Diagram

/-- The layout `to_drawing` gives a diagram: its layers composed by the loop
of `Drawing.then`, the identity on its domain when it has none. -/
def layout (f : Diagram S) : Layout :=
  match f.layers with
  | [] => idLayout f.dom.length 1
  | l :: ls => (ls.map Layer.layout).foldl compose l.layout

theorem foldl_compose_fit (ls : List Layout) {L : Layout} (hL : Fit L)
    (h : ∀ M ∈ ls, Fit M) : Fit (ls.foldl compose L) := by
  induction ls generalizing L with
  | nil => exact hL
  | cons M ls ih =>
    exact ih (compose_fit hL (h M (by simp))) fun M' hM' => h M' (by simp [hM'])

/-- The layout of every diagram is plane, in its frame, its boundary wires in
order. -/
theorem layout_fit (f : Diagram S) : Fit f.layout := by
  unfold layout
  split
  · exact idLayout_fit _ (by norm_num)
  · refine foldl_compose_fit _ (Layer.layout_fit _) ?_
    simp only [List.mem_map]
    rintro _ ⟨l, _, rfl⟩; exact l.layout_fit

end Diagram

end Premonoidal
