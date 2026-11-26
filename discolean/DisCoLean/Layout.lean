import DisCoLean.Layout.Model
import Mathlib.Data.Rat.Defs
import Mathlib.Order.MinMax
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum.Basic
import Mathlib.Tactic.Linarith

/-!
# The `make_space` layout of `to_drawing` is plane

`Diagram.to_drawing` draws a diagram layer by layer: a layer is
`Drawing.from_box` of its box between `Drawing.id` of the wires on each side,
put side by side by `Drawing.tensor`, and `>>` is `Drawing.then`, which runs
`make_space` on one half for each wire until the outputs of the top are above
the inputs of the bottom, stacks them, then recentres the outputs of the box
it added, by `align_box_cod` when it has as many outputs as inputs and by
`reposition_box_cod` otherwise. `make_space(s, x0)` shifts by `s` every node
of one half with `x ≥ x0`, the map `shiftX x0 s`.

The model is exact for boxes with no drawing attributes and wires whose labels
fit in a unit: `Row` holds the `x` of every node of a layer, `Layer.row` is
where `from_box`, `id` and `tensor` put them, `compose` is `Drawing.then` with
its loop and its recentering, and `Diagram.layout` is `to_drawing` of a
diagram with at least one layer. Each
layer is one high, the boxes and the identities beside them alike, so
`tensor` never calls `stretch` with a non-zero height, which `stretch` returns
unchanged. `test_layout` of `discolean/test_premonoidal.py` checks every
node, the width and the height against `to_drawing` on generated diagrams,
through the `oracle`.

* `Row.Good`: the wires of a row are in order, those beside the box more than a
  quarter from its nodes; `Row.picture_fit` says its picture is plane.
* `Row.fresh_good`, `Row.fresh_centred`: the row of `from_box` between two
  identities is good, with the room to recentre its outputs.
* `shift_expands`: `make_space` never brings two points closer, so it keeps a
  row good, and `Row.Centred`, the room its outputs have once recentred.
* `align_good`: the loop of `Drawing.then` ends with each output of the top
  above the input of the bottom with the same index.
* `Diagram.layout_good`: the layout of every diagram of Diagram(S) is a stack of
  good rows, each starting where the one above it ends, so
  `Diagram.layout_plane`: its picture is plane.
-/

namespace MakeSpace

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

/-- A picture: its wires and boxes, the `x` of the wires on its top boundary
(`dom`) and bottom boundary (`cod`), its width and height. -/
structure Picture where
  wires : List Wire
  rects : List Rect
  dom : List ℚ
  cod : List ℚ
  width : ℚ
  height : ℚ

/-- A plane picture: no two wires meet but at an end, no two boxes overlap and
no wire passes through a box. -/
structure Plane (L : Picture) : Prop where
  wires : L.wires.Pairwise WireApart
  rects : L.rects.Pairwise RectApart
  wireRect : ∀ w ∈ L.wires, ∀ b ∈ L.rects, WireRect w b

/-- A picture lies in its frame, a quarter away from its left and right
sides, and its boundary wires are in order from left to right. -/
structure Bounded (L : Picture) : Prop where
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

theorem shift_sorted {x0 s : ℚ} (hs : 0 ≤ s) {l : List ℚ} (h : l.Pairwise (· < ·)) :
    (l.map (shiftX x0 s)).Pairwise (· < ·) :=
  List.pairwise_map.2 (h.imp (shift_strict hs))


/-! ## Stacking and side by side -/

def Wire.up (d : ℚ) (w : Wire) : Wire := { w with lo := w.lo + d, hi := w.hi + d }
def Rect.up (d : ℚ) (b : Rect) : Rect := { b with y := b.y + d }


/-- `L` on top of `M`, as `Drawing.then` stacks them once aligned. -/
def Picture.then (L M : Picture) : Picture where
  wires := L.wires.map (Wire.up M.height) ++ M.wires
  rects := L.rects.map (Rect.up M.height) ++ M.rects
  dom := L.dom
  cod := M.cod
  width := max L.width M.width
  height := L.height + M.height

/-- Stacking two plane pictures is plane: the one lies above the other. -/
theorem then_plane {L M : Picture} (hL : Plane L) (hM : Plane M)
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
    simp only [Picture.then, List.mem_append, List.mem_map]
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

theorem then_bounded {L M : Picture} (bL : Bounded L) (bM : Bounded M) :
    Bounded (L.then M) := by
  have wL := le_max_left L.width M.width
  have wM := le_max_right L.width M.width
  have hL := bL.height
  have hM := bM.height
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, bL.dom_sorted, bM.cod_sorted, by
    simp only [Picture.then]; linarith, by simp only [Picture.then]; linarith [bL.width]⟩
  · simp only [Picture.then, List.mem_append, List.mem_map]
    rintro _ (⟨w, hw, rfl⟩ | hw)
    · exact ⟨(bL.wire_x w hw).1, by simp only [Wire.up]; linarith [(bL.wire_x w hw).2]⟩
    · exact ⟨(bM.wire_x _ hw).1, by linarith [(bM.wire_x _ hw).2]⟩
  · simp only [Picture.then, List.mem_append, List.mem_map]
    rintro _ (⟨w, hw, rfl⟩ | hw)
    · obtain ⟨h0, h1, h2⟩ := bL.wire_y w hw
      simp only [Wire.up]; exact ⟨by linarith, by linarith, by linarith⟩
    · obtain ⟨h0, h1, h2⟩ := bM.wire_y _ hw
      exact ⟨h0, h1, by linarith⟩
  · simp only [Picture.then, List.mem_append, List.mem_map]
    rintro _ (⟨b, hb, rfl⟩ | hb)
    · exact ⟨by simpa [Rect.up] using (bL.rect_x b hb).1, fun a ha => by
        have := (bL.rect_x b hb).2 a (by simpa [Rect.up] using ha)
        exact ⟨this.1, by linarith [this.2]⟩⟩
    · exact ⟨(bM.rect_x _ hb).1, fun a ha => by
        have := (bM.rect_x _ hb).2 a ha
        exact ⟨this.1, by linarith [this.2]⟩⟩
  · simp only [Picture.then, List.mem_append, List.mem_map]
    rintro _ (⟨b, hb, rfl⟩ | hb)
    · obtain ⟨h0, h1, h2⟩ := bL.rect_y b hb
      simp only [Rect.up]; exact ⟨h0, by linarith, by linarith⟩
    · obtain ⟨h0, h1, h2⟩ := bM.rect_y _ hb
      exact ⟨h0, h1, by linarith⟩
  · intro a ha; exact ⟨(bL.dom a ha).1, by simp only [Picture.then]; linarith [(bL.dom a ha).2]⟩
  · intro a ha; exact ⟨(bM.cod a ha).1, by simp only [Picture.then]; linarith [(bM.cod a ha).2]⟩


theorem shift_mono {x0 s : ℚ} (hs : 0 ≤ s) : Monotone (shiftX x0 s) := fun a b h => by
  have := shift_expands (x0 := x0) hs h; linarith

/-! ## Rows -/

namespace Row

@[simp] theorem top_map (φ : ℚ → ℚ) (r : Row) : (r.map φ).top = r.top.map φ := by
  simp [top, map]

@[simp] theorem bot_map (φ : ℚ → ℚ) (r : Row) : (r.map φ).bot = r.bot.map φ := by
  simp [bot, map]

@[simp] theorem hull_map (φ : ℚ → ℚ) (r : Row) : (r.map φ).hull = r.hull.map φ := by
  simp [hull, map]

@[simp] theorem left_map (φ : ℚ → ℚ) (r : Row) : (r.map φ).left = r.left.map φ := rfl
@[simp] theorem right_map (φ : ℚ → ℚ) (r : Row) : (r.map φ).right = r.right.map φ := rfl
@[simp] theorem dom_map (φ : ℚ → ℚ) (r : Row) : (r.map φ).dom = r.dom.map φ := rfl
@[simp] theorem cod_map (φ : ℚ → ℚ) (r : Row) : (r.map φ).cod = r.cod.map φ := rfl
@[simp] theorem box_map (φ : ℚ → ℚ) (r : Row) : (r.map φ).box = φ r.box := rfl

/-- A good row in a frame `w` wide: its wires in order at its top and at its
bottom, those beside the box more than a quarter from its nodes, and every
node a quarter inside the frame. -/
structure Good (r : Row) (w : ℚ) : Prop where
  top : r.top.Pairwise (· < ·)
  bot : r.bot.Pairwise (· < ·)
  left : ∀ l ∈ r.left, ∀ a ∈ r.hull, l + 1 / 4 < a
  right : ∀ a ∈ r.hull, ∀ q ∈ r.right, a + 1 / 4 < q
  inside : ∀ a ∈ r.top ++ r.bot ++ r.hull, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ w

theorem Good.mono {r : Row} {w w' : ℚ} (h : r.Good w) (hw : w ≤ w') : r.Good w' :=
  ⟨h.top, h.bot, h.left, h.right, fun a ha => ⟨(h.inside a ha).1, by linarith [(h.inside a ha).2]⟩⟩

/-- `make_space` keeps a row good, in a frame `s` wider. -/
theorem Good.shift {r : Row} {w x0 s : ℚ} (hs : 0 ≤ s) (h : r.Good w) :
    (r.map (shiftX x0 s)).Good (w + s) where
  top := by rw [top_map]; exact shift_sorted hs h.top
  bot := by rw [bot_map]; exact shift_sorted hs h.bot
  left := by
    simp only [left_map, hull_map, List.mem_map]
    rintro _ ⟨l, hl, rfl⟩ _ ⟨a, ha, rfl⟩
    exact shift_lt hs (by norm_num) (h.left l hl a ha)
  right := by
    simp only [right_map, hull_map, List.mem_map]
    rintro _ ⟨a, ha, rfl⟩ _ ⟨q, hq, rfl⟩
    exact shift_lt hs (by norm_num) (h.right a ha q hq)
  inside := by
    simp only [top_map, bot_map, hull_map, ← List.map_append, List.mem_map]
    rintro _ ⟨a, ha, rfl⟩
    exact ⟨le_trans (h.inside a ha).1 (le_shift hs),
      by linarith [(h.inside a ha).2, shift_le_add (x0 := x0) (a := a) hs]⟩

/-! ## Recentering -/

theorem foldr_min_le {b : ℚ} : ∀ {xs : List ℚ}, ∀ a ∈ b :: xs, xs.foldr min b ≤ a
  | [], a, h => by simp at h; simp [h]
  | x :: xs, a, h => by
    simp only [List.foldr_cons]
    rcases List.mem_cons.1 h with rfl | h
    · exact le_trans (min_le_right _ _) (foldr_min_le a (by simp))
    rcases List.mem_cons.1 h with rfl | h
    · exact min_le_left _ _
    · exact le_trans (min_le_right _ _) (foldr_min_le a (List.mem_cons_of_mem _ h))

theorem le_foldr_max {b : ℚ} : ∀ {xs : List ℚ}, ∀ a ∈ b :: xs, a ≤ xs.foldr max b
  | [], a, h => by simp at h; simp [h]
  | x :: xs, a, h => by
    simp only [List.foldr_cons]
    rcases List.mem_cons.1 h with rfl | h
    · exact le_trans (le_foldr_max a (by simp)) (le_max_right _ _)
    rcases List.mem_cons.1 h with rfl | h
    · exact le_max_left _ _
    · exact le_trans (le_foldr_max a (List.mem_cons_of_mem _ h)) (le_max_right _ _)

theorem foldr_min_mem {b : ℚ} : ∀ {xs : List ℚ}, xs.foldr min b ∈ b :: xs
  | [] => by simp
  | x :: xs => by
    simp only [List.foldr_cons]
    rcases min_choice x (xs.foldr min b) with h | h <;> rw [h]
    · simp
    · rcases List.mem_cons.1 (foldr_min_mem (b := b) (xs := xs)) with h' | h' <;> simp [h']

theorem foldr_max_mem {b : ℚ} : ∀ {xs : List ℚ}, xs.foldr max b ∈ b :: xs
  | [] => by simp
  | x :: xs => by
    simp only [List.foldr_cons]
    rcases max_choice x (xs.foldr max b) with h | h <;> rw [h]
    · simp
    · rcases List.mem_cons.1 (foldr_max_mem (b := b) (xs := xs)) with h' | h' <;> simp [h']

theorem foldr_min_map {φ : ℚ → ℚ} (hφ : Monotone φ) (b : ℚ) :
    ∀ xs : List ℚ, (xs.map φ).foldr min (φ b) = φ (xs.foldr min b)
  | [] => rfl
  | x :: xs => by simp only [List.map_cons, List.foldr_cons, foldr_min_map hφ b xs, hφ.map_min]

theorem foldr_max_map {φ : ℚ → ℚ} (hφ : Monotone φ) (b : ℚ) :
    ∀ xs : List ℚ, (xs.map φ).foldr max (φ b) = φ (xs.foldr max b)
  | [] => rfl
  | x :: xs => by simp only [List.map_cons, List.foldr_cons, foldr_max_map hφ b xs, hφ.map_max]

theorem lo_mem_hull (r : Row) : r.lo ∈ r.hull := by
  rcases List.mem_cons.1 (foldr_min_mem (b := r.box) (xs := r.dom)) with h | h <;>
    simp [lo, hull, h]

theorem hi_mem_hull (r : Row) : r.hi ∈ r.hull := by
  rcases List.mem_cons.1 (foldr_max_mem (b := r.box) (xs := r.dom)) with h | h <;>
    simp [hi, hull, h]

theorem lo_le_hi (r : Row) : r.lo ≤ r.hi :=
  le_trans (foldr_min_le _ (List.mem_cons_self ..)) (le_foldr_max _ (List.mem_cons_self ..))

/-- Room to recentre the outputs: centred on the middle of `lo` and `hi`, they
are more than a quarter from the wires on either side and inside the frame. -/
structure Centred (r : Row) (w : ℚ) : Prop where
  left : ∀ l ∈ r.left, l + 1 / 4 < (r.lo + r.hi - ((r.cod.length : ℚ) - 1)) / 2
  right : ∀ q ∈ r.right, (r.lo + r.hi + ((r.cod.length : ℚ) - 1)) / 2 + 1 / 4 < q
  low : 1 / 4 ≤ (r.lo + r.hi - ((r.cod.length : ℚ) - 1)) / 2
  high : (r.lo + r.hi + ((r.cod.length : ℚ) - 1)) / 2 + 1 / 4 ≤ w

/-- `make_space` keeps the room to recentre: the middle of `lo` and `hi`
moves away from the wires on either side at least as much as they do. -/
theorem Centred.shift {r : Row} {w x0 s : ℚ} (hs : 0 ≤ s) (g : r.Good w) (h : r.Centred w) :
    (r.map (shiftX x0 s)).Centred (w + s) := by
  have elo : (r.map (shiftX x0 s)).lo = shiftX x0 s r.lo := foldr_min_map (shift_mono hs) _ _
  have ehi : (r.map (shiftX x0 s)).hi = shiftX x0 s r.hi := foldr_max_map (shift_mono hs) _ _
  have len : (r.map (shiftX x0 s)).cod.length = r.cod.length := by simp
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [elo, ehi, len]
  · simp only [left_map, List.mem_map]
    rintro _ ⟨l, hl, rfl⟩
    have a := shift_expands (x0 := x0) hs (le_of_lt (lt_of_le_of_lt (by norm_num : l ≤ l + 1 / 4)
      (g.left l hl _ r.lo_mem_hull)))
    have b := shift_expands (x0 := x0) hs (le_of_lt (lt_of_le_of_lt (by norm_num : l ≤ l + 1 / 4)
      (g.left l hl _ r.hi_mem_hull)))
    linarith [h.left l hl]
  · simp only [right_map, List.mem_map]
    rintro _ ⟨q, hq, rfl⟩
    have a := shift_expands (x0 := x0) hs (le_of_lt (lt_of_le_of_lt (by linarith : r.lo ≤ r.lo + 1 / 4)
      (g.right _ r.lo_mem_hull q hq)))
    have b := shift_expands (x0 := x0) hs (le_of_lt (lt_of_le_of_lt (by linarith : r.hi ≤ r.hi + 1 / 4)
      (g.right _ r.hi_mem_hull q hq)))
    linarith [h.right q hq]
  · linarith [h.low, le_shift (x0 := x0) (a := r.lo) hs, le_shift (x0 := x0) (a := r.hi) hs]
  · linarith [h.high, shift_le_add (x0 := x0) (a := r.lo) hs, shift_le_add (x0 := x0) (a := r.hi) hs]

theorem mem_centred {r : Row} {a : ℚ} (h : a ∈ r.centred) :
    (r.lo + r.hi - ((r.cod.length : ℚ) - 1)) / 2 ≤ a
    ∧ a ≤ (r.lo + r.hi + ((r.cod.length : ℚ) - 1)) / 2 := by
  simp only [centred, List.mem_map, List.mem_range] at h
  obtain ⟨k, hk, rfl⟩ := h
  have : (k : ℚ) + 1 ≤ r.cod.length := by exact_mod_cast hk
  have : (0 : ℚ) ≤ k := Nat.cast_nonneg k
  constructor <;> linarith

theorem centred_sorted (r : Row) : r.centred.Pairwise (· < ·) := by
  unfold centred
  refine List.pairwise_map.2 (List.pairwise_lt_range.imp fun {a b} h => ?_)
  have : (a : ℚ) < b := by exact_mod_cast h
  linarith

theorem top_sorted_parts {r : Row} (h : r.top.Pairwise (· < ·)) :
    r.left.Pairwise (· < ·) ∧ r.right.Pairwise (· < ·) ∧ ∀ l ∈ r.left, ∀ q ∈ r.right, l < q := by
  unfold top at h
  obtain ⟨h1, h2, h3⟩ := List.pairwise_append.1 h
  obtain ⟨h4, _, _⟩ := List.pairwise_append.1 h1
  exact ⟨h4, h2, fun l hl q hq => h3 l (List.mem_append_left _ hl) q hq⟩

theorem mem_nodes {r : Row} {a : ℚ} : a ∈ r.top ++ r.bot ++ r.hull ↔
    a ∈ r.left ∨ a ∈ r.dom ∨ a ∈ r.cod ∨ a ∈ r.right ∨ a = r.box := by
  simp only [top, bot, hull, List.mem_append, List.mem_singleton]
  tauto

theorem Good.node {r : Row} {w : ℚ} (g : r.Good w) {a : ℚ}
    (h : a ∈ r.left ∨ a ∈ r.dom ∨ a ∈ r.cod ∨ a ∈ r.right ∨ a = r.box) :
    1 / 4 ≤ a ∧ a + 1 / 4 ≤ w := g.inside a (mem_nodes.2 h)

/-- Recentering a good row with room to do it gives a good row. -/
theorem Good.recentre {r : Row} {w : ℚ} (g : r.Good w) (h : r.Centred w) :
    r.recentre.Good w := by
  unfold Row.recentre
  split_ifs with hn
  · refine ⟨g.top, g.top, ?_, ?_, ?_⟩
    · intro l hl a ha
      simp only [hull, List.mem_append, List.mem_singleton] at ha
      exact g.left l hl a (by rcases ha with (ha | ha) | rfl <;> first | simp [hull, ha] | simp [hull])
    · intro a ha q hq
      simp only [hull, List.mem_append, List.mem_singleton] at ha
      exact g.right a (by rcases ha with (ha | ha) | rfl <;> first | simp [hull, ha] | simp [hull]) q hq
    · intro a ha
      rcases mem_nodes.1 ha with h | h | h | h | h
      · exact g.node (.inl h)
      · exact g.node (.inr (.inl h))
      · exact g.node (.inr (.inl h))
      · exact g.node (.inr (.inr (.inr (.inl h))))
      · exact g.node (.inr (.inr (.inr (.inr h))))
  · obtain ⟨sl, sr, lr⟩ := top_sorted_parts g.top
    have bot : (r.left ++ r.centred ++ r.right).Pairwise (· < ·) := by
      refine List.pairwise_append.2 ⟨List.pairwise_append.2 ⟨sl, centred_sorted r, ?_⟩, sr, ?_⟩
      · intro l hl a ha; linarith [h.left l hl, (mem_centred ha).1]
      · intro a ha q hq
        rcases List.mem_append.1 ha with ha | ha
        · exact lr a ha q hq
        · linarith [h.right q hq, (mem_centred ha).2]
    refine ⟨g.top, bot, ?_, ?_, ?_⟩
    · intro l hl a ha
      simp only [hull, List.mem_append, List.mem_singleton] at ha
      rcases ha with (ha | ha) | rfl
      · exact g.left l hl a (by simp [hull, ha])
      · linarith [h.left l hl, (mem_centred ha).1]
      · exact g.left l hl _ (by simp [hull])
    · intro a ha q hq
      simp only [hull, List.mem_append, List.mem_singleton] at ha
      rcases ha with (ha | ha) | rfl
      · exact g.right a (by simp [hull, ha]) q hq
      · linarith [h.right q hq, (mem_centred ha).2]
      · exact g.right _ (by simp [hull]) q hq
    · intro a ha
      rcases mem_nodes.1 ha with h' | h' | h' | h' | h'
      · exact g.node (.inl h')
      · exact g.node (.inr (.inl h'))
      · exact ⟨by linarith [h.low, (mem_centred h').1], by linarith [h.high, (mem_centred h').2]⟩
      · exact g.node (.inr (.inr (.inr (.inl h'))))
      · exact g.node (.inr (.inr (.inr (.inr h'))))

theorem recentre_top (r : Row) : r.recentre.top = r.top := by
  unfold recentre; split_ifs <;> rfl

theorem recentre_bot_length (r : Row) : r.recentre.bot.length = r.bot.length := by
  unfold recentre; split_ifs with h <;> simp [bot, centred, h]

end Row


/-! ## The rows `to_drawing` starts from -/

theorem foldr_min_ge {c b : ℚ} (hb : c ≤ b) :
    ∀ {xs : List ℚ}, (∀ x ∈ xs, c ≤ x) → c ≤ xs.foldr min b
  | [], _ => hb
  | x :: xs, h => le_min (h x (by simp)) (foldr_min_ge hb fun y hy => h y (by simp [hy]))

theorem foldr_max_le {c b : ℚ} (hb : b ≤ c) :
    ∀ {xs : List ℚ}, (∀ x ∈ xs, x ≤ c) → xs.foldr max b ≤ c
  | [], _ => hb
  | x :: xs, h => max_le (h x (by simp)) (foldr_max_le hb fun y hy => h y (by simp [hy]))

theorem mem_seq {a x : ℚ} {n : ℕ} (h : x ∈ (List.range n).map fun k : ℕ => a + (k : ℚ)) :
    a ≤ x ∧ x + 1 ≤ a + n := by
  simp only [List.mem_map, List.mem_range] at h
  obtain ⟨k, hk, rfl⟩ := h
  have : (k : ℚ) + 1 ≤ n := by exact_mod_cast hk
  have : (0 : ℚ) ≤ k := Nat.cast_nonneg k
  constructor <;> linarith

theorem seq_sorted (a : ℚ) (n : ℕ) :
    ((List.range n).map fun k : ℕ => a + (k : ℚ)).Pairwise (· < ·) :=
  List.pairwise_map.2 (List.pairwise_lt_range.imp fun {i j} h => by
    have : (i : ℚ) < j := by exact_mod_cast h
    linarith)

theorem seq_first {a : ℚ} {n : ℕ} (hn : 0 < n) :
    a ∈ (List.range n).map fun k : ℕ => a + (k : ℚ) :=
  List.mem_map.2 ⟨0, List.mem_range.2 hn, by simp⟩

theorem seq_last {a : ℚ} {n : ℕ} (hn : 0 < n) :
    a + n - 1 ∈ (List.range n).map fun k : ℕ => a + (k : ℚ) :=
  List.mem_map.2 ⟨n - 1, List.mem_range.2 (by omega), by
    rw [Nat.cast_sub (by omega : 1 ≤ n), Nat.cast_one]; ring⟩

namespace Row

theorem fresh_left (cl n m cr : ℕ) : (fresh cl n m cr).left
    = (List.range cl).map fun k : ℕ => 1 / 4 + (k : ℚ) := rfl
theorem fresh_dom (cl n m cr : ℕ) : (fresh cl n m cr).dom
    = (List.range n).map fun k : ℕ => start cl + (content n m - n) / 2 + 1 / 2 + (k : ℚ) := rfl
theorem fresh_box (cl n m cr : ℕ) : (fresh cl n m cr).box = start cl + content n m / 2 := rfl
theorem fresh_cod (cl n m cr : ℕ) : (fresh cl n m cr).cod
    = (List.range m).map fun k : ℕ => start cl + (content n m - m) / 2 + 1 / 2 + (k : ℚ) := rfl
theorem fresh_right (cl n m cr : ℕ) : (fresh cl n m cr).right
    = (List.range cr).map fun k : ℕ => start cl + content n m + 1 / 2 + (k : ℚ) := rfl

theorem content_ge (n m : ℕ) : (n : ℚ) ≤ content n m ∧ (m : ℚ) ≤ content n m ∧ 1 ≤ content n m :=
  ⟨le_trans (le_max_left _ _) (le_max_left _ _), le_trans (le_max_right _ _) (le_max_left _ _),
    le_max_right _ _⟩

theorem start_nonneg (cl : ℕ) : 0 ≤ start cl := by
  unfold start; split_ifs with h
  · exact le_rfl
  · have : (1 : ℚ) ≤ cl := Nat.one_le_cast.2 (Nat.one_le_iff_ne_zero.2 h)
    linarith

theorem left_le_start {cl : ℕ} {l : ℚ}
    (h : l ∈ (List.range cl).map fun k : ℕ => 1 / 4 + (k : ℚ)) : l + 1 / 2 ≤ start cl := by
  obtain ⟨h1, h2⟩ := mem_seq h
  unfold start; split_ifs with h0
  · rw [h0, Nat.cast_zero] at h2; linarith
  · linarith

theorem width_ge (cl n m cr : ℕ) : start cl + content n m ≤ freshWidth cl n m cr := by
  unfold freshWidth; split_ifs with h
  · exact le_rfl
  · have : (1 : ℚ) ≤ cr := Nat.one_le_cast.2 (Nat.one_le_iff_ne_zero.2 h)
    linarith

theorem right_le_width {cl n m cr : ℕ} {q : ℚ} (h : q ∈ (fresh cl n m cr).right) :
    q + 1 / 4 ≤ freshWidth cl n m cr := by
  rw [fresh_right] at h
  obtain ⟨h1, h2⟩ := mem_seq h
  unfold freshWidth; split_ifs with h0
  · rw [h0, Nat.cast_zero] at h2; linarith
  · linarith

/-- A row is good when its wires are sorted, its box between `A` and `B` and
the wires beside it more than a quarter away, all in the frame. -/
theorem good_of_bounds {r : Row} {w A B : ℚ}
    (sl : r.left.Pairwise (· < ·)) (sd : r.dom.Pairwise (· < ·))
    (sc : r.cod.Pairwise (· < ·)) (sr : r.right.Pairwise (· < ·))
    (hd : ∀ p ∈ r.dom, A ≤ p ∧ p ≤ B) (hc : ∀ p ∈ r.cod, A ≤ p ∧ p ≤ B)
    (hb : A ≤ r.box ∧ r.box ≤ B) (hA : 1 / 4 ≤ A) (hB : B + 1 / 4 ≤ w)
    (hl : ∀ l ∈ r.left, 1 / 4 ≤ l ∧ l + 1 / 4 < A)
    (hr : ∀ q ∈ r.right, B + 1 / 4 < q ∧ q + 1 / 4 ≤ w) : r.Good w := by
  have hh : ∀ a ∈ r.hull, A ≤ a ∧ a ≤ B := by
    intro a ha
    simp only [hull, List.mem_append, List.mem_singleton] at ha
    rcases ha with (ha | ha) | rfl
    · exact hd a ha
    · exact hc a ha
    · exact hb
  have sorted : ∀ {mid : List ℚ}, mid.Pairwise (· < ·) → (∀ p ∈ mid, A ≤ p ∧ p ≤ B) →
      (r.left ++ mid ++ r.right).Pairwise (· < ·) := by
    intro mid sm hm
    refine List.pairwise_append.2 ⟨List.pairwise_append.2 ⟨sl, sm, fun l hl' p hp => ?_⟩, sr, ?_⟩
    · linarith [(hl l hl').2, (hm p hp).1]
    · intro a ha q hq
      rcases List.mem_append.1 ha with ha | ha
      · linarith [(hl a ha).2, (hr q hq).1]
      · linarith [(hm a ha).2, (hr q hq).1]
  refine ⟨sorted sd hd, sorted sc hc, fun l hl' a ha => ?_, fun a ha q hq => ?_, fun a ha => ?_⟩
  · linarith [(hl l hl').2, (hh a ha).1]
  · linarith [(hh a ha).2, (hr q hq).1]
  · rcases mem_nodes.1 ha with h | h | h | h | h
    · exact ⟨(hl a h).1, by linarith [(hl a h).2]⟩
    · exact ⟨by linarith [(hd a h).1], by linarith [(hd a h).2]⟩
    · exact ⟨by linarith [(hc a h).1], by linarith [(hc a h).2]⟩
    · exact ⟨by linarith [(hr a h).1], (hr a h).2⟩
    · subst h; exact ⟨by linarith [hb.1], by linarith [hb.2]⟩

/-- `Drawing.from_box` between two identities is a good row. -/
theorem fresh_good (cl n m cr : ℕ) : (fresh cl n m cr).Good (freshWidth cl n m cr) := by
  obtain ⟨cn, cm, c1⟩ := content_ge n m
  have x0 := start_nonneg cl
  have wg := width_ge cl n m cr
  apply good_of_bounds (A := start cl + 1 / 2) (B := start cl + content n m - 1 / 2)
  · rw [fresh_left]; exact seq_sorted _ _
  · rw [fresh_dom]; exact seq_sorted _ _
  · rw [fresh_cod]; exact seq_sorted _ _
  · rw [fresh_right]; exact seq_sorted _ _
  · rw [fresh_dom]; intro p hp; have := mem_seq hp; constructor <;> linarith [this.1, this.2]
  · rw [fresh_cod]; intro p hp; have := mem_seq hp; constructor <;> linarith [this.1, this.2]
  · rw [fresh_box]; constructor <;> linarith
  · linarith
  · linarith
  · rw [fresh_left]; intro l hl
    have := mem_seq hl
    have := left_le_start hl
    constructor <;> linarith
  · intro q hq
    have := right_le_width hq
    rw [fresh_right] at hq
    have := mem_seq hq
    constructor <;> linarith

theorem seq_lo_hi {a b : ℚ} {n : ℕ} (hn : 0 < n) (hb1 : a ≤ b) (hb2 : b ≤ a + n - 1) :
    ((List.range n).map fun k : ℕ => a + (k : ℚ)).foldr min b = a
    ∧ ((List.range n).map fun k : ℕ => a + (k : ℚ)).foldr max b = a + n - 1 := by
  constructor
  · refine le_antisymm (foldr_min_le _ (List.mem_cons_of_mem _ (seq_first hn))) ?_
    exact foldr_min_ge hb1 fun x hx => (mem_seq hx).1
  · refine le_antisymm ?_ (le_foldr_max _ (List.mem_cons_of_mem _ (seq_last hn)))
    exact foldr_max_le hb2 fun x hx => by linarith [(mem_seq hx).2]

/-- Its outputs are already where `reposition_box_cod` puts them, so it has
the room to recentre. -/
theorem fresh_centred (cl n m cr : ℕ) : (fresh cl n m cr).Centred (freshWidth cl n m cr) := by
  obtain ⟨cn, cm, c1⟩ := content_ge n m
  have x0 := start_nonneg cl
  have wg := width_ge cl n m cr
  have sum : (fresh cl n m cr).lo + (fresh cl n m cr).hi = 2 * start cl + content n m := by
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp only [lo, hi, fresh_dom, fresh_box, List.range_zero, List.map_nil, List.foldr_nil]
      ring
    · have hnq : (1 : ℚ) ≤ n := Nat.one_le_cast.2 hn
      obtain ⟨e1, e2⟩ := seq_lo_hi (a := start cl + (content n m - n) / 2 + 1 / 2)
        (b := start cl + content n m / 2) hn (by linarith) (by linarith)
      simp only [lo, hi, fresh_dom, fresh_box]
      rw [e1, e2]; ring
  have hm : ((fresh cl n m cr).cod.length : ℚ) = m := by
    rw [fresh_cod, List.length_map, List.length_range]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [hm, sum]
  · intro l hl
    rw [fresh_left] at hl
    have := left_le_start hl
    linarith
  · intro q hq
    rw [fresh_right] at hq
    have := mem_seq hq
    linarith [this.1]
  · linarith
  · linarith

theorem Good.top_pos {r : Row} {w : ℚ} (g : r.Good w) : ∀ x ∈ r.top, 1 / 4 ≤ x :=
  fun x hx => (g.inside x (List.mem_append_left _ (List.mem_append_left _ hx))).1

theorem Good.bot_pos {r : Row} {w : ℚ} (g : r.Good w) : ∀ x ∈ r.bot, 1 / 4 ≤ x :=
  fun x hx => (g.inside x (List.mem_append_left _ (List.mem_append_right _ hx))).1

/-! ## The picture of a row -/

/-- The picture of a row, one high: the wires beside its box from top to
bottom, those into it from the top to its top side, those out of it from its
bottom side to the bottom, and its box. -/
def picture (w : ℚ) (r : Row) : Picture where
  wires := (r.left ++ r.right).map (fun x => ⟨x, 0, 1⟩) ++ r.dom.map (fun x => ⟨x, 3 / 4, 1⟩)
    ++ r.cod.map (fun x => ⟨x, 0, 1 / 4⟩)
  rects := [⟨r.hull, 1 / 2, 1 / 4⟩]
  dom := r.top
  cod := r.bot
  width := w
  height := 1

/-- The picture of a good row is plane and in its frame. -/
theorem picture_fit {r : Row} {w : ℚ} (g : r.Good w) :
    Plane (r.picture w) ∧ Bounded (r.picture w) := by
  have ht : (r.left ++ r.dom ++ r.right).Pairwise (· < ·) := g.top
  have hb : (r.left ++ r.cod ++ r.right).Pairwise (· < ·) := g.bot
  obtain ⟨sl, sr, lr⟩ := top_sorted_parts g.top
  have sd := (List.pairwise_append.1 (List.pairwise_append.1 ht).1).2.1
  have sc := (List.pairwise_append.1 (List.pairwise_append.1 hb).1).2.1
  have dh : ∀ p ∈ r.dom, p ∈ r.hull := fun p hp => by simp [hull, hp]
  have ch : ∀ p ∈ r.cod, p ∈ r.hull := fun p hp => by simp [hull, hp]
  have lrh : ∀ a, a ∈ r.left ∨ a ∈ r.right → ∀ p ∈ r.hull, a < p ∨ p < a := by
    rintro a (ha | ha) p hp
    · exact .inl (by linarith [g.left a ha p hp])
    · exact .inr (by linarith [g.right p hp a ha])
  have hull_node : ∀ a ∈ r.hull, 1 / 4 ≤ a ∧ a + 1 / 4 ≤ w := by
    intro a ha
    simp only [hull, List.mem_append, List.mem_singleton] at ha
    rcases ha with (ha | ha) | ha
    · exact g.node (.inr (.inl ha))
    · exact g.node (.inr (.inr (.inl ha)))
    · exact g.node (.inr (.inr (.inr (.inr ha))))
  have box := g.node (a := r.box) (.inr (.inr (.inr (.inr rfl))))
  refine ⟨⟨?_, List.pairwise_singleton _ _, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, g.top, g.bot,
    by norm_num [picture], by simp only [picture]; linarith [box.1, box.2]⟩⟩
  · show List.Pairwise WireApart (((r.left ++ r.right).map (fun x => (⟨x, 0, 1⟩ : Wire))
      ++ r.dom.map (fun x => (⟨x, 3 / 4, 1⟩ : Wire))) ++ r.cod.map (fun x => (⟨x, 0, 1 / 4⟩ : Wire)))
    refine List.pairwise_append.2 ⟨List.pairwise_append.2 ⟨?_, ?_, ?_⟩, ?_, ?_⟩
    · exact List.pairwise_map.2 ((List.pairwise_append.2 ⟨sl, sr, lr⟩).imp fun h => .inl h)
    · exact List.pairwise_map.2 (sd.imp fun h => .inl h)
    · simp only [List.mem_map, List.mem_append]
      rintro _ ⟨a, ha, rfl⟩ _ ⟨p, hp, rfl⟩
      rcases lrh a ha p (dh p hp) with h | h
      · exact .inl h
      · exact .inr (.inl h)
    · exact List.pairwise_map.2 (sc.imp fun h => .inl h)
    · simp only [List.mem_append, List.mem_map]
      rintro _ (⟨a, ha, rfl⟩ | ⟨p, hp, rfl⟩) _ ⟨q, hq, rfl⟩
      · rcases lrh a ha q (ch q hq) with h | h
        · exact .inl h
        · exact .inr (.inl h)
      · exact .inr (.inr (.inr (by norm_num)))
  · intro w' hw b hb
    simp only [picture, List.mem_singleton] at hb
    subst hb
    simp only [picture, List.mem_append, List.mem_map] at hw
    rcases hw with ((⟨a, ha | ha, rfl⟩ | ⟨p, hp, rfl⟩) | ⟨p, hp, rfl⟩)
    · exact .inl fun x hx => g.left a ha x hx
    · exact .inr (.inl fun x hx => g.right x hx a ha)
    · exact .inr (.inr (.inr (by norm_num)))
    · exact .inr (.inr (.inl (by norm_num)))
  · intro w' hw
    simp only [picture, List.mem_append, List.mem_map] at hw
    rcases hw with ((⟨a, ha | ha, rfl⟩ | ⟨p, hp, rfl⟩) | ⟨p, hp, rfl⟩)
    · exact g.node (.inl ha)
    · exact g.node (.inr (.inr (.inr (.inl ha))))
    · exact hull_node p (dh p hp)
    · exact hull_node p (ch p hp)
  · intro w' hw
    simp only [picture, List.mem_append, List.mem_map] at hw
    rcases hw with ((⟨a, -, rfl⟩ | ⟨p, -, rfl⟩) | ⟨p, -, rfl⟩) <;> norm_num [picture]
  · intro b hb
    simp only [picture, List.mem_singleton] at hb
    subst hb
    exact ⟨by simp [hull], hull_node⟩
  · intro b hb
    simp only [picture, List.mem_singleton] at hb
    subst hb; norm_num [picture]
  · intro a ha
    simp only [picture, top, List.mem_append] at ha
    rcases ha with (ha | ha) | ha
    · exact g.node (.inl ha)
    · exact g.node (.inr (.inl ha))
    · exact g.node (.inr (.inr (.inr (.inl ha))))
  · intro a ha
    simp only [picture, bot, List.mem_append] at ha
    rcases ha with (ha | ha) | ha
    · exact g.node (.inl ha)
    · exact g.node (.inr (.inr (.inl ha)))
    · exact g.node (.inr (.inr (.inr (.inl ha))))

end Row

/-- The picture of nothing, `w` wide. -/
def Picture.empty (w : ℚ) : Picture := ⟨[], [], [], [], w, 0⟩

theorem Picture.empty_fit {w : ℚ} (hw : 0 ≤ w) :
    Plane (Picture.empty w) ∧ Bounded (Picture.empty w) :=
  ⟨⟨.nil, .nil, by simp [Picture.empty]⟩,
   ⟨by simp [Picture.empty], by simp [Picture.empty], by simp [Picture.empty],
    by simp [Picture.empty], by simp [Picture.empty], by simp [Picture.empty], .nil, .nil,
    le_rfl, hw⟩⟩

theorem pictures_fit {w : ℚ} (hw : 0 ≤ w) : ∀ rs : List Row, (∀ r ∈ rs, r.Good w) →
    Plane (rs.foldr (fun r P => (r.picture w).then P) (Picture.empty w))
    ∧ Bounded (rs.foldr (fun r P => (r.picture w).then P) (Picture.empty w))
  | [], _ => Picture.empty_fit hw
  | r :: rs, h =>
    have ih := pictures_fit hw rs fun s hs => h s (List.mem_cons_of_mem _ hs)
    have g := Row.picture_fit (h r List.mem_cons_self)
    ⟨then_plane g.1 ih.1 g.2 ih.2, then_bounded g.2 ih.2⟩

/-- The picture of a stack: its rows from top to bottom. -/
def Stack.picture (T : Stack) : Picture :=
  T.rows.foldr (fun r P => (r.picture T.width).then P) (Picture.empty T.width)

/-! ## Stacks -/

/-- The rows meet: each starts with the wires the one above it ends with. -/
def Meet : List Row → Prop
  | [] => True
  | [_] => True
  | r :: s :: rs => r.bot = s.top ∧ Meet (s :: rs)

theorem meet_map (φ : ℚ → ℚ) : ∀ {l : List Row}, Meet l → Meet (l.map (Row.map φ))
  | [], _ => trivial
  | [_], _ => trivial
  | _ :: s :: rs, h => ⟨by rw [Row.bot_map, Row.top_map, h.1], meet_map φ (l := s :: rs) h.2⟩

theorem meet_snoc {y : Row} : ∀ {l : List Row}, Meet l →
    (∀ x, l.getLast? = some x → x.bot = y.top) → Meet (l ++ [y])
  | [], _, _ => trivial
  | [x], _, h => ⟨h x rfl, trivial⟩
  | _ :: s :: rs, h, h' =>
    ⟨h.1, meet_snoc (l := s :: rs) h.2 fun x hx => h' x (by rw [List.getLast?_cons_cons]; exact hx)⟩

/-- A good stack: its rows good in its width, each meeting the next. -/
structure Stack.Good (T : Stack) : Prop where
  rows : ∀ r ∈ T.rows, r.Good T.width
  meet : Meet T.rows

theorem Stack.last_mem (T : Stack) : T.last ∈ T.rows := by simp [Stack.rows]

theorem Stack.rows_shift (x0 s : ℚ) (T : Stack) :
    (T.shift x0 s).rows = T.rows.map (Row.map (shiftX x0 s)) := by
  simp [Stack.rows, Stack.shift]

/-- `make_space` keeps a stack good. -/
theorem Stack.Good.shift {T : Stack} {x0 s : ℚ} (hs : 0 ≤ s) (h : T.Good) :
    (T.shift x0 s).Good where
  rows := by
    rw [Stack.rows_shift]
    simp only [List.mem_map]
    rintro _ ⟨r, hr, rfl⟩
    exact (h.rows r hr).shift hs
  meet := by rw [Stack.rows_shift]; exact meet_map _ h.meet

/-- The picture of a good stack is plane and in its frame. -/
theorem Stack.Good.picture_fit {T : Stack} (h : T.Good) :
    Plane T.picture ∧ Bounded T.picture := by
  have box := (h.rows _ (Stack.last_mem T)).node (a := T.last.box) (.inr (.inr (.inr (.inr rfl))))
  exact pictures_fit (by linarith [box.1, box.2]) T.rows h.rows

/-! ## The loop of `Drawing.then` -/

/-- The invariant of the loop: the stack good, the row it adds good and with
the room to recentre. -/
structure StateGood (p : State) : Prop where
  stack : p.1.Good
  row : p.2.1.Good p.2.2
  centred : p.2.1.Centred p.2.2

theorem alignAt_cases (i : ℕ) (p : State) : alignAt i p = p ∨ ∃ x0 s, 0 ≤ s ∧
    (alignAt i p = (p.1, p.2.1.map (shiftX x0 s), p.2.2 + s)
      ∨ alignAt i p = (p.1.shift x0 s, p.2)) := by
  unfold alignAt; dsimp only
  by_cases h : p.2.1.top.getD i 0 < p.1.last.bot.getD i 0
  · rw [ite_t h]; exact .inr ⟨_, _, by linarith, .inl rfl⟩
  by_cases h' : p.1.last.bot.getD i 0 < p.2.1.top.getD i 0
  · rw [ite_f h, ite_t h']; exact .inr ⟨_, _, by linarith, .inr rfl⟩
  · rw [ite_f h, ite_f h']; exact .inl rfl

theorem alignAt_good (i : ℕ) {p : State} (h : StateGood p) : StateGood (alignAt i p) := by
  rcases alignAt_cases i p with e | ⟨x0, s, hs, e | e⟩ <;> rw [e]
  · exact h
  · exact ⟨h.stack, h.row.shift hs, h.centred.shift hs h.row⟩
  · exact ⟨h.stack.shift hs, h.row, h.centred⟩

theorem alignAt_length (i : ℕ) (p : State) :
    (alignAt i p).1.last.bot.length = p.1.last.bot.length
    ∧ (alignAt i p).2.1.top.length = p.2.1.top.length := by
  rcases alignAt_cases i p with e | ⟨x0, s, hs, e | e⟩ <;> rw [e] <;> simp [Stack.shift]

theorem alignAt_bot (i : ℕ) (p : State) : (alignAt i p).2.1.bot.length = p.2.1.bot.length := by
  rcases alignAt_cases i p with e | ⟨x0, s, hs, e | e⟩ <;> simp [e]

theorem foldl_alignAt_bot : ∀ (is : List ℕ) (p : State),
    (is.foldl (fun p i => alignAt i p) p).2.1.bot.length = p.2.1.bot.length
  | [], _ => rfl
  | i :: is, p => (foldl_alignAt_bot is (alignAt i p)).trans (alignAt_bot i p)

/-- A turn that shifts `b` from its wire `k` by the distance to `a`'s keeps
the wires before `k` and lines up wire `k`. -/
theorem shift_prefix {a b : List ℚ} {k : ℕ} (hka : k < a.length) (hkb : k < b.length)
    (sb : b.Pairwise (· < ·)) (pb : ∀ x ∈ b, 1 / 4 ≤ x) (hlt : b[k] < a[k])
    (h : ∀ j < k, a[j]? = b[j]?) :
    ∀ j < k + 1, a[j]? = (b.map (shiftX (if k = 0 then 0 else b[k]) (a[k] - b[k])))[j]? := by
  have pk := pb _ (List.getElem_mem hkb)
  intro j hj
  rw [List.getElem?_map]
  rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
  · have hjb : j < b.length := by omega
    have lt := List.pairwise_iff_getElem.1 sb j k hjb hkb hj
    rw [h j hj, List.getElem?_eq_getElem hjb]
    simp only [Option.map_some]
    unfold shiftX
    rw [ite_f (by rw [ite_f (show ¬ k = 0 by omega)]; exact not_le.2 lt)]
  · rw [List.getElem?_eq_getElem hka, List.getElem?_eq_getElem hkb]
    simp only [Option.map_some]
    unfold shiftX
    rw [ite_t (by split_ifs <;> linarith)]
    congr 1; ring

theorem alignAt_step {p : State} {k : ℕ} (hkc : k < p.1.last.bot.length)
    (hkd : k < p.2.1.top.length)
    (h1 : p.1.last.bot.Pairwise (· < ·)) (h2 : p.2.1.top.Pairwise (· < ·))
    (q1 : ∀ x ∈ p.1.last.bot, 1 / 4 ≤ x) (q2 : ∀ x ∈ p.2.1.top, 1 / 4 ≤ x)
    (h : ∀ j < k, p.1.last.bot[j]? = p.2.1.top[j]?) :
    ∀ j < k + 1, (alignAt k p).1.last.bot[j]? = (alignAt k p).2.1.top[j]? := by
  have etop : p.1.last.bot.getD k 0 = p.1.last.bot[k] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkc]; rfl
  have ebot : p.2.1.top.getD k 0 = p.2.1.top[k] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkd]; rfl
  unfold alignAt; dsimp only; rw [etop, ebot]
  by_cases c1 : p.2.1.top[k] < p.1.last.bot[k]
  · rw [ite_t c1]
    simp only [Row.top_map]
    exact shift_prefix hkc hkd h2 q2 c1 h
  by_cases c2 : p.1.last.bot[k] < p.2.1.top[k]
  · rw [ite_f c1, ite_t c2]
    simp only [Stack.shift, Row.bot_map]
    intro j hj
    exact (shift_prefix hkd hkc h1 q1 c2 (fun j hj => (h j hj).symm) j hj).symm
  · rw [ite_f c1, ite_f c2]
    intro j hj
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · exact h j hj
    · rw [List.getElem?_eq_getElem hkc, List.getElem?_eq_getElem hkd]
      congr 1; linarith [not_lt.1 c1, not_lt.1 c2]

theorem align_inv {T : Stack} {r : Row} {w : ℚ} (h : StateGood (T, r, w))
    (hn : T.last.bot.length = r.top.length) :
    ∀ k ≤ r.top.length,
      let p := (List.range k).foldl (fun p i => alignAt i p) (T, r, w)
      StateGood p ∧ p.1.last.bot.length = r.top.length ∧ p.2.1.top.length = r.top.length
      ∧ ∀ j < k, p.1.last.bot[j]? = p.2.1.top[j]? := by
  intro k hk
  induction k with
  | zero => exact ⟨h, hn, rfl, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
  | succ k ih =>
    obtain ⟨g, hc, hd, hp⟩ := ih (by omega)
    simp only [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    have len := alignAt_length k ((List.range k).foldl (fun p i => alignAt i p) (T, r, w))
    have last := g.stack.rows _ (Stack.last_mem _)
    exact ⟨alignAt_good k g, len.1.trans hc, len.2.trans hd,
      alignAt_step (by omega) (by omega) last.bot g.row.top last.bot_pos g.row.top_pos hp⟩

/-- The loop of `Drawing.then` keeps its invariant and ends with every output
of the stack above the input of the row with the same index. -/
theorem align_good {T : Stack} {r : Row} {w : ℚ} (h : StateGood (T, r, w))
    (hn : T.last.bot.length = r.top.length) :
    StateGood (align T r w) ∧ (align T r w).1.last.bot = (align T r w).2.1.top := by
  unfold align
  obtain ⟨g, hc, hd, hp⟩ := align_inv h hn r.top.length le_rfl
  refine ⟨g, List.ext_getElem? fun j => ?_⟩
  by_cases hj : j < r.top.length
  · exact hp j hj
  · rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by omega)]

/-- `Drawing.then` of a good stack and a good row with room to recentre is a
good stack. -/
theorem compose_good {T : Stack} {r : Row} {w : ℚ} (hT : T.Good) (g : r.Good w)
    (c : r.Centred w) (hn : T.last.bot.length = r.top.length) : (compose T r w).Good := by
  obtain ⟨⟨hs, hr, hc⟩, hm⟩ := align_good ⟨hT, g, c⟩ hn
  have hw1 := le_max_left (align T r w).1.width (align T r w).2.2
  have hw2 := le_max_right (align T r w).1.width (align T r w).2.2
  have hl : (align T r w).1.rows.getLast? = some (align T r w).1.last := by simp [Stack.rows]
  refine ⟨?_, meet_snoc hs.meet fun x hx => ?_⟩
  · intro x hx
    simp only [compose, Stack.rows, List.mem_append, List.mem_singleton] at hx
    rcases hx with (hx | rfl) | rfl
    · exact (hs.rows x (by simp [Stack.rows, hx])).mono hw1
    · exact (hs.rows _ (Stack.last_mem _)).mono hw1
    · exact (hr.recentre hc).mono hw2
  · have hx' : (align T r w).1.rows.getLast? = some x := hx
    rw [hl] at hx'
    obtain rfl := Option.some.inj hx'
    show _ = (align T r w).2.1.recentre.top
    rw [Row.recentre_top]; exact hm

theorem compose_bot (T : Stack) (r : Row) (w : ℚ) :
    (compose T r w).last.bot.length = r.bot.length := by
  show (align T r w).2.1.recentre.bot.length = r.bot.length
  rw [Row.recentre_bot_length]
  exact foldl_alignAt_bot _ _

end MakeSpace

namespace Premonoidal

open MakeSpace

variable {S : Signature}

theorem Layer.top_row (l : Layer S) : l.row.top.length = l.dom.length := by
  simp only [Layer.row, Row.fresh, Row.top, Layer.dom, List.length_append, List.length_map,
    List.length_range]

theorem Layer.bot_row (l : Layer S) : l.row.bot.length = l.cod.length := by
  simp only [Layer.row, Row.fresh, Row.bot, Layer.cod, List.length_append, List.length_map,
    List.length_range]

theorem Layer.row_good (l : Layer S) : l.row.Good l.width := Row.fresh_good _ _ _ _

theorem Layer.row_centred (l : Layer S) : l.row.Centred l.width := Row.fresh_centred _ _ _ _

theorem layoutFrom_good : ∀ (ls : List (Layer S)) {T : Stack} {X : List S.Ob},
    T.Good → T.last.bot.length = X.length → Composable X ls → (layoutFrom T ls).Good
  | [], _, _, h, _, _ => h
  | l :: ls, T, X, h, hl, hc =>
    layoutFrom_good ls (compose_good h l.row_good l.row_centred
        (by rw [hl, Layer.top_row, hc.1]))
      (by rw [compose_bot, Layer.bot_row]) hc.2

theorem Layer.start_good (l : Layer S) : (⟨[], l.row, l.width⟩ : Stack).Good :=
  ⟨fun x hx => by simp [Stack.rows] at hx; subst hx; exact l.row_good, trivial⟩

/-- The layout of every diagram is a good stack: its rows good and meeting. -/
theorem Diagram.layout_good (f : Diagram S) {T : Stack} (h : f.layout = some T) : T.Good := by
  obtain ⟨dom, layers, hc⟩ := f
  cases layers with
  | nil => exact absurd h (by simp [Diagram.layout])
  | cons l ls =>
    have h' : layoutFrom ⟨[], l.row, l.width⟩ ls = T := Option.some.inj h
    subst h'
    exact layoutFrom_good ls l.start_good (Layer.bot_row l) hc.2

/-- **The layout of every diagram is plane**: no two wires meet but at an end,
no two boxes overlap, no wire passes through a box, all in its frame. -/
theorem Diagram.layout_plane (f : Diagram S) {T : Stack} (h : f.layout = some T) :
    Plane T.picture ∧ Bounded T.picture := (f.layout_good h).picture_fit

end Premonoidal
