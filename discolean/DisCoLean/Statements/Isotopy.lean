import DisCoLean.Isotopy

/-!
# What is proved of tweening the drawing of an interchanger

The model of `Drawing.interpolate` on two boxes, see `DisCoLean/Isotopy.lean`:
it is not linked to the Python by a test, the positions of the
counterexample are those `to_drawing` gives on `amp/isotopy-gif`.
`Audit.lean` covers this file.
-/

namespace Statements.Isotopy
open _root_.Isotopy

/-- Two boxes separated along the same side in both drawings stay apart for
every `t` in `[0, 1]`.
Rules out: a tween that moves a side of a box by anything but a convex
combination of its two positions. -/
theorem stays_apart {b b' : Box m} {c c' : Box n} {t : ℚ}
    (ht : 0 ≤ t ∧ t ≤ 1) (hb : b.half = b'.half) (hc : c.half = c'.half)
    (h : LeftOf b c ∧ LeftOf b' c' ∨ LeftOf c b ∧ LeftOf c' b'
      ∨ Below b c ∧ Below b' c' ∨ Below c b ∧ Below c' b') :
    Apart (b.lerp t b') (c.lerp t c') := _root_.Isotopy.stays_apart ht hb hc h

/-- A horizontal move that keeps the left-to-right order of two boxes on one
row, or keeps them on different rows, keeps them apart: the first and last
phases of `interpolate` across an interchanger.
Rules out: a phase that moves a box vertically while claiming to slide it. -/
theorem slides_apart {b b' : Box m} {c c' : Box n} {t : ℚ}
    (ht : 0 ≤ t ∧ t ≤ 1) (hb : b.half = b'.half) (hc : c.half = c'.half)
    (hby : b.y = b'.y) (hcy : c.y = c'.y)
    (h : LeftOf b c ∧ LeftOf b' c' ∨ LeftOf c b ∧ LeftOf c' b'
      ∨ Below b c ∨ Below c b) :
    Apart (b.lerp t b') (c.lerp t c') :=
  _root_.Isotopy.slides_apart ht hb hc hby hcy h

/-- The tween of `f @ x @ x >> x @ x @ x @ g` to its interchanger
`x @ g >> f @ x`, with `f : x → x ** 3` and `g : x ** 2 → x`, overlaps the
boxes at `t = 21/40`: both drawings are plane, separated along different sides.
Rules out: the claim that the straight-line tween of every interchanger is
a planar isotopy. -/
theorem interchanger_overlaps :
    LeftOf f₀ g₀ ∧ ¬ LeftOf f₁ g₁ ∧ Below g₀ f₀ ∧ Below f₁ g₁
    ∧ ¬ Apart (f₀.lerp (21/40) f₁) (g₀.lerp (21/40) g₁) :=
  _root_.Isotopy.interchanger_overlaps

end Statements.Isotopy
