import DisCoLean.Layout

/-!
# What is proved of the `make_space` layout of `to_drawing`

The model of `Diagram.to_drawing`, see `DisCoLean/Layout/Model.lean`: a
`Row` is the `x` of every node of a layer as `from_box`, `id` and `tensor`
place them, `compose` is `Drawing.then` with its loop of `make_space` and the
recentering of `align_box_cod` and `reposition_box_cod`, and `stretch` is the
identity since every layer is one high. `test_layout` of
`discolean/test_premonoidal.py` checks every node, the width and the height
against the Python, through the `oracle`. `Audit.lean` covers this file.
-/

namespace Statements.Layout
open _root_.MakeSpace

/-- `make_space(s, x0)` never brings two points closer.
Rules out: a `make_space` that shifts the nodes left of `x0` and not those
right of it. -/
theorem shift_expands {x0 s a b : ℚ} (hs : 0 ≤ s) (h : a ≤ b) :
    b - a ≤ shiftX x0 s b - shiftX x0 s a := _root_.MakeSpace.shift_expands hs h

/-- The row of `Drawing.from_box` between two identities is good and has the
room to recentre its outputs.
Rules out: a `tensor` with no gap after the wires left of the box. -/
theorem fresh {cl n m cr : ℕ} :
    (Row.fresh cl n m cr).Good (Row.freshWidth cl n m cr)
    ∧ (Row.fresh cl n m cr).Centred (Row.freshWidth cl n m cr) :=
  ⟨Row.fresh_good cl n m cr, Row.fresh_centred cl n m cr⟩

/-- Recentering the outputs of a good row with the room for it keeps it good.
Rules out: centring the outputs on the box node alone, which can push them
past the wires beside it. -/
theorem recentre {r : Row} {w : ℚ} (g : r.Good w) (c : r.Centred w) : r.recentre.Good w :=
  g.recentre c

/-- The picture of a good row is plane and in its frame.
Rules out: a row whose box hull reaches the wires beside it. -/
theorem picture_fit {r : Row} {w : ℚ} (g : r.Good w) :
    Plane (r.picture w) ∧ Bounded (r.picture w) := Row.picture_fit g

/-- The loop of `Drawing.then` keeps the stack and the row good and ends with
every output of the stack above the input of the row with the same index.
Rules out: shifting from `0` on every wire, which moves the wires already
aligned. -/
theorem align {T : Stack} {r : Row} {w : ℚ} (h : StateGood (T, r, w))
    (hn : T.last.bot.length = r.top.length) :
    StateGood (_root_.MakeSpace.align T r w)
    ∧ (_root_.MakeSpace.align T r w).1.last.bot = (_root_.MakeSpace.align T r w).2.1.top :=
  align_good h hn

/-- **The layout of every diagram is plane**: `to_drawing` of every diagram
of Diagram(S) with at least one layer is a stack of good rows, each starting
where the one above it ends, and its picture is plane and in its frame: no two
wires meet but at an end, no two boxes overlap, no wire passes through a box.
Rules out: a layout that draws a whiskered box over the wires beside it. -/
theorem layout_plane {S : Premonoidal.Signature} (f : Premonoidal.Diagram S) {T : Stack}
    (h : f.layout = some T) : T.Good ∧ Plane T.picture ∧ Bounded T.picture :=
  ⟨f.layout_good h, f.layout_plane h⟩

end Statements.Layout
