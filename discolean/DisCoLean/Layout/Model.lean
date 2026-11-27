import DisCoLean.Premonoidal.Diagram

/-!
# The model of `to_drawing`, executable

The definitions of `DisCoLean/Layout.lean`, with no Mathlib so that the
`oracle` runs them: where `Diagram.to_drawing` puts the `x` of every node of a
diagram with one box per layer, boxes with no drawing attributes and wires
whose labels fit in a unit.
-/

namespace MakeSpace

/-- The map of `make_space(s, x0)` on the `x` of a node. -/
def shiftX (x0 s x : Rat) : Rat := if x0 ≤ x then x + s else x

/-- A layer of a drawing, one high, as `to_drawing` places it: the `x` of the
wires on its left, of the input ports of its box, of its box node, of its
output ports and of the wires on its right. -/
structure Row where
  left : List Rat
  dom : List Rat
  box : Rat
  cod : List Rat
  right : List Rat

namespace Row

/-- The wires at the top of a row, from left to right. -/
def top (r : Row) : List Rat := r.left ++ r.dom ++ r.right

/-- The wires at the bottom of a row. -/
def bot (r : Row) : List Rat := r.left ++ r.cod ++ r.right

/-- The nodes of its box: `add_box_corners` draws their hull, a quarter wider. -/
def hull (r : Row) : List Rat := r.dom ++ r.cod ++ [r.box]

/-- A map on the `x` of every node. -/
def map (φ : Rat → Rat) (r : Row) : Row :=
  ⟨r.left.map φ, r.dom.map φ, φ r.box, r.cod.map φ, r.right.map φ⟩

/-- The least of the inputs and the box node, `left` in `reposition_box_cod`. -/
def lo (r : Row) : Rat := r.dom.foldr min r.box

/-- The greatest of them, `right` there. -/
def hi (r : Row) : Rat := r.dom.foldr max r.box

/-- The outputs centred one apart on the middle of `lo` and `hi`. -/
def centred (r : Row) : List Rat :=
  (List.range r.cod.length).map fun k : Nat =>
    (r.lo + r.hi - ((r.cod.length : Rat) - 1)) / 2 + k

/-- What `Drawing.then` does to the box it adds: `align_box_cod`, each output
under its input, when it has as many of each, else `reposition_box_cod`. -/
def recentre (r : Row) : Row :=
  if r.dom.length = r.cod.length then { r with cod := r.dom } else { r with cod := r.centred }

/-- The width of a box with `n` inputs and `m` outputs, `content` in
`Drawing.from_box`. -/
def content (n m : Nat) : Rat := max (max (n : Rat) (m : Rat)) 1

/-- Where the box starts: after the `Drawing.id` of `cl` wires, `cl - 1/2`
wide, and the quarter `Drawing.tensor` adds after a whiskered drawing. -/
def start (cl : Nat) : Rat := if cl = 0 then 0 else (cl : Rat) - 1 / 4

/-- `Drawing.id(left) @ Drawing.from_box(box) @ Drawing.id(right)`, for `cl`
wires on the left, a box with `n` inputs and `m` outputs and `cr` wires on
the right. -/
def fresh (cl n m cr : Nat) : Row :=
  let xb := start cl
  let c := content n m
  ⟨(List.range cl).map fun k : Nat => 1 / 4 + (k : Rat),
   (List.range n).map fun k : Nat => xb + (c - n) / 2 + 1 / 2 + k,
   xb + c / 2,
   (List.range m).map fun k : Nat => xb + (c - m) / 2 + 1 / 2 + k,
   (List.range cr).map fun k : Nat => xb + c + 1 / 2 + k⟩

/-- Its width. -/
def freshWidth (cl n m cr : Nat) : Rat :=
  if cr = 0 then start cl + content n m else start cl + content n m + cr - 1 / 4

end Row

/-- A stack of rows: those above the last one, the last one and the width. -/
structure Stack where
  above : List Row
  last : Row
  width : Rat

/-- The rows from top to bottom. -/
def Stack.rows (S : Stack) : List Row := S.above ++ [S.last]

/-- `make_space` on the top half of `Drawing.then`. -/
def Stack.shift (x0 s : Rat) (S : Stack) : Stack :=
  ⟨S.above.map (Row.map (shiftX x0 s)), S.last.map (shiftX x0 s), S.width + s⟩

/-- The state of the loop of `Drawing.then`: the top half, the row it adds
and its width. -/
abbrev State := Stack × Row × Rat

/-- One turn of the loop, on wire `i`: the half whose end of that wire is left
of the other's is shifted by `make_space` from that end, from `0` for the first
wire, by the distance between them. -/
def alignAt (i : Nat) (p : State) : State :=
  let top := p.1.last.bot.getD i 0
  let bot := p.2.1.top.getD i 0
  if bot < top then
    (p.1, p.2.1.map (shiftX (if i = 0 then 0 else bot) (top - bot)), p.2.2 + (top - bot))
  else if top < bot then (p.1.shift (if i = 0 then 0 else top) (bot - top), p.2)
  else p

/-- The loop of `Drawing.then`, over the inputs of the row it adds. -/
def align (S : Stack) (r : Row) (w : Rat) : State :=
  (List.range r.top.length).foldl (fun p i => alignAt i p) (S, r, w)

/-- `Drawing.then` of a stack and a row: aligned, stacked, and the row's
outputs recentred. -/
def compose (S : Stack) (r : Row) (w : Rat) : Stack :=
  let p := align S r w
  ⟨p.1.above ++ [p.1.last], p.2.1.recentre, max p.1.width p.2.2⟩

end MakeSpace

namespace Premonoidal

open MakeSpace

variable {S : Signature}

/-- The row of a layer when it is drawn on its own. -/
def Layer.row (l : Layer S) : Row :=
  Row.fresh l.left.length (S.dom l.box).length (S.cod l.box).length l.right.length

/-- Its width. -/
def Layer.width (l : Layer S) : Rat :=
  Row.freshWidth l.left.length (S.dom l.box).length (S.cod l.box).length l.right.length

/-- Composing a stack with the layers below it, one at a time. -/
def layoutFrom (T : Stack) : List (Layer S) → Stack
  | [] => T
  | l :: ls => layoutFrom (compose T l.row l.width) ls

/-- The layout `to_drawing` gives a diagram with at least one layer. -/
def Diagram.layout (f : Diagram S) : Option Stack :=
  match f.layers with
  | [] => none
  | l :: ls => some (layoutFrom ⟨[], l.row, l.width⟩ ls)

end Premonoidal
