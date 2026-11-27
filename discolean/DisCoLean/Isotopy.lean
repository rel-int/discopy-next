import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum.Basic

/-!
# Tweening the drawing of an interchanger

`Drawing.interpolate(other, t)` moves each node of a drawing a fraction `t`
along the segment to its position in `other`. A box is drawn by
`add_box_corners` as the rectangle around its box node and its wire nodes,
a quarter wider on each side, and half a wire less than its height high. This
file models one box as the x of those nodes and the y of its box node, and
asks when the tween of two drawings keeps two boxes apart.

`stays_apart`: two boxes separated along one side in both drawings, e.g. the
one left of the other, stay separated along it for every `t` in `[0, 1]`,
since each side of a rectangle moves by a convex combination.

`interchanger_overlaps`: the drawings `to_drawing` gives of
`f @ x @ x >> x @ x @ x @ g` and of its interchanger `x @ g >> f @ x`, with
`f : x → x ** 3` and `g : x ** 2 → x`, are separated along different sides,
and halfway through, at `t = 21/40`, the boxes overlap: the straight-line
tween of one interchanger is not always a planar isotopy.
-/

namespace Isotopy

/-- The point a fraction `t` of the way from `a` to `b`. -/
def lerp (t a b : ℚ) : ℚ := a + t * (b - a)

/-- A box in a drawing: the x of its box node and wire nodes, the y of its
box node and half its height, i.e. `max(height / 2 - 1/4, 1/4)`. -/
structure Box (n : ℕ) where
  xs : Fin n → ℚ
  y : ℚ
  half : ℚ

/-- The box a fraction `t` of the way from `b` to `b'`, the same box drawn
twice: its height does not move. -/
def Box.lerp (t : ℚ) (b b' : Box n) : Box n :=
  ⟨fun i => Isotopy.lerp t (b.xs i) (b'.xs i), Isotopy.lerp t b.y b'.y, b.half⟩

/-- `b` is left of `c`: the right side of the one, its rightmost node and a
quarter, is at most the left side of the other. -/
def LeftOf (b : Box m) (c : Box n) : Prop := ∀ i j, b.xs i + 1 / 2 ≤ c.xs j

/-- `b` is below `c`: the y axis points up, as in `to_drawing`. -/
def Below (b : Box m) (c : Box n) : Prop := b.y + b.half ≤ c.y - c.half

/-- Two boxes are apart when one is left of or below the other: their
rectangles meet at most on a side. -/
def Apart (b : Box m) (c : Box n) : Prop :=
  LeftOf b c ∨ LeftOf c b ∨ Below b c ∨ Below c b

theorem lerp_le {t a a' c c' : ℚ} (ht : 0 ≤ t ∧ t ≤ 1)
    (h : a ≤ c) (h' : a' ≤ c') : lerp t a a' ≤ lerp t c c' := by
  unfold lerp
  nlinarith [mul_nonneg (sub_nonneg.2 ht.2) (sub_nonneg.2 h),
    mul_nonneg ht.1 (sub_nonneg.2 h')]

theorem leftOf_lerp {b b' : Box m} {c c' : Box n} {t : ℚ}
    (ht : 0 ≤ t ∧ t ≤ 1) (h : LeftOf b c) (h' : LeftOf b' c') :
    LeftOf (b.lerp t b') (c.lerp t c') := fun i j => by
  have := lerp_le ht (h i j) (h' i j)
  simp only [Box.lerp, lerp] at this ⊢
  linarith

theorem below_lerp {b b' : Box m} {c c' : Box n} {t : ℚ}
    (ht : 0 ≤ t ∧ t ≤ 1) (hb : b.half = b'.half) (hc : c.half = c'.half)
    (h : Below b c) (h' : Below b' c') :
    Below (b.lerp t b') (c.lerp t c') := by
  unfold Below at h h'
  have := lerp_le ht h (show b'.y + b.half ≤ c'.y - c.half by rw [hb, hc]; exact h')
  simp only [Below, Box.lerp, lerp] at this ⊢
  linarith

/-- Two boxes separated along the same side in two drawings of them stay
apart along the tween. -/
theorem stays_apart {b b' : Box m} {c c' : Box n} {t : ℚ}
    (ht : 0 ≤ t ∧ t ≤ 1) (hb : b.half = b'.half) (hc : c.half = c'.half)
    (h : LeftOf b c ∧ LeftOf b' c' ∨ LeftOf c b ∧ LeftOf c' b'
      ∨ Below b c ∧ Below b' c' ∨ Below c b ∧ Below c' b') :
    Apart (b.lerp t b') (c.lerp t c') := by
  rcases h with h | h | h | h
  · exact .inl (leftOf_lerp ht h.1 h.2)
  · exact .inr (.inl (leftOf_lerp ht h.1 h.2))
  · exact .inr (.inr (.inl (below_lerp ht hb hc h.1 h.2)))
  · exact .inr (.inr (.inr (below_lerp ht hc hb h.1 h.2)))

/-- A horizontal move keeps two boxes apart when it keeps the left-to-right
order of boxes on one row: boxes on different rows stay on them, since their
y does not move. These are the first and last phases of `interpolate` across
an interchanger, the middle one being `stays_apart` on boxes left of each
other. -/
theorem slides_apart {b b' : Box m} {c c' : Box n} {t : ℚ}
    (ht : 0 ≤ t ∧ t ≤ 1) (hb : b.half = b'.half) (hc : c.half = c'.half)
    (hby : b.y = b'.y) (hcy : c.y = c'.y)
    (h : LeftOf b c ∧ LeftOf b' c' ∨ LeftOf c b ∧ LeftOf c' b'
      ∨ Below b c ∨ Below c b) :
    Apart (b.lerp t b') (c.lerp t c') := by
  apply stays_apart ht hb hc
  rcases h with h | h | h | h
  · exact .inl h
  · exact .inr (.inl h)
  · refine .inr (.inr (.inl ⟨h, ?_⟩))
    unfold Below at h ⊢; rw [← hby, ← hcy, ← hb, ← hc]; exact h
  · refine .inr (.inr (.inr ⟨h, ?_⟩))
    unfold Below at h ⊢; rw [← hby, ← hcy, ← hb, ← hc]; exact h

/-! ## The counterexample

The nodes of each box are its box node, its `box_dom` then its `box_cod`
nodes, at the positions `to_drawing` gives them on `amp/isotopy-gif`. -/

/-- `f : x → x ** 3` in `f @ x @ x >> x @ x @ x @ g`, on top. -/
def f₀ : Box 5 := ⟨![3/2, 3/2, 1/2, 3/2, 5/2], 3/2, 1/4⟩
/-- `g : x ** 2 → x` in `f @ x @ x >> x @ x @ x @ g`, at the bottom right. -/
def g₀ : Box 4 := ⟨![4, 7/2, 9/2, 4], 1/2, 1/4⟩
/-- `f` in `x @ g >> f @ x`, at the bottom. -/
def f₁ : Box 5 := ⟨![3/2, 3/2, 1/2, 3/2, 5/2], 1/2, 1/4⟩
/-- `g` in `x @ g >> f @ x`, on top, partly above `f`. -/
def g₁ : Box 4 := ⟨![7/2, 5/2, 4, 7/2], 3/2, 1/4⟩

/-- The two drawings are plane, but along different sides: `g` is right of
`f` only in the first one, and below it in the first one, above in the other.
Halfway through the tween, they overlap. -/
theorem interchanger_overlaps :
    LeftOf f₀ g₀ ∧ ¬ LeftOf f₁ g₁ ∧ Below g₀ f₀ ∧ Below f₁ g₁
    ∧ ¬ Apart (f₀.lerp (21/40) f₁) (g₀.lerp (21/40) g₁) := by
  simp only [Apart, LeftOf, Below, Box.lerp, lerp, f₀, g₀, f₁, g₁,
    Fin.forall_fin_succ, Fin.isValue, Matrix.cons_val_zero,
    Matrix.cons_val_succ, not_or]
  norm_num

end Isotopy
