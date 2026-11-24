import DisCoLean.Layout

/-!
# What is proved of the `make_space` layout of `to_drawing`

The model of `Diagram.to_drawing`, see `DisCoLean/Layout.lean`: wires are
vertical segments and boxes the rectangles around their nodes, `@` puts two
layouts side by side, `>>` runs the loop of `Drawing.then`, `make_space` on
one half for each wire, then stacks them. It leaves out `stretch`, every layer
being one high, and the recentering of `reposition_box_cod` and
`align_box_cod`. It is not linked to the Python by a test. `Audit.lean`
covers this file.
-/

namespace Statements.Layout
open _root_.MakeSpace

/-- `make_space(s, x0)` never brings two points closer.
Rules out: a `make_space` that shifts the nodes left of `x0` and not those
right of it. -/
theorem shift_expands {x0 s a b : ℚ} (hs : 0 ≤ s) (h : a ≤ b) :
    b - a ≤ shiftX x0 s b - shiftX x0 s a := _root_.MakeSpace.shift_expands hs h

/-- So it keeps a layout plane and in its frame.
Rules out: a shift by a negative amount, which can push a box onto a wire. -/
theorem shift_fit {x0 s : ℚ} (hs : 0 ≤ s) {L : Layout} (h : Fit L) : Fit (L.shift x0 s) :=
  ⟨shift_plane hs h.1, shift_bounded hs h.2⟩

/-- Two plane layouts side by side, `gap ≥ 0` apart and of one height, are
plane: the one lies left of the other.
Rules out: putting the second at the width of the first minus its margin. -/
theorem tensor_fit {L M : Layout} {gap : ℚ} (hgap : 0 ≤ gap) (heq : L.height = M.height)
    (hL : Fit L) (hM : Fit M) : Fit (L.tensor M gap) :=
  ⟨tensor_plane hgap hL.1 hM.1 hL.2 hM.2, tensor_bounded hgap heq hL.2 hM.2⟩

/-- The loop of `Drawing.then` puts every output of the top layout above the
input of the bottom one with the same index, and composing keeps the layout
plane.
Rules out: shifting from `0` on every wire, which moves the wires already
aligned. -/
theorem compose {L M : Layout} (hn : L.cod.length = M.dom.length) (hL : Fit L) (hM : Fit M) :
    (align L M).1.cod = (align L M).2.dom ∧ Fit (_root_.MakeSpace.compose L M) :=
  ⟨align_aligned hn hL hM, compose_fit hL hM⟩

/-- **The layout of every diagram is plane**: no two wires meet but at an
end, no two boxes overlap, no wire passes through a box, and the boundary
wires are in order.
Rules out: a layout that draws a whiskered box over the wires beside it. -/
theorem layout_fit {S : Premonoidal.Signature} (f : Premonoidal.Diagram S) : Fit f.layout :=
  f.layout_fit

end Statements.Layout
