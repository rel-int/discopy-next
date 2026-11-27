import DisCoLean.Drawing

/-!
# What is proved of the layered drawing of a diagram

The model of drawing with no optimisation, see `DisCoLean/Drawing.lean`: wire
`k` of each row at `x = 2 * k + 1`, one strip per layer. It is not linked to
`to_drawing` by a test, whose layout `make_space` moves. `Audit.lean` covers
this file.
-/

namespace Statements.Drawing
open _root_.Premonoidal

variable {S : Signature}

/-- Every strip of the drawing of a diagram is plane: no two wires cross, the
wires keep their order, none meets the box, and the ports are on the box.
Rules out: a layout that puts the wires right of a box at the `x` they had
before it, on top of the box when its codomain is longer than its domain. -/
theorem plane (f : Diagram S) : ∀ s ∈ f.drawing, s.Plane := f.drawing_plane

/-- The strips meet, from the domain down: each starts with as many wires as
the one above it ends with.
Rules out: drawing the right wires of a layer from its codomain on top. -/
theorem meets (f : Diagram S) : Meets f.dom.length f.drawing := f.drawing_meets

/-- The drawing is a functor of strict premonoidal categories: identities are
no strip, a box one strip, composition stacks the strips and whiskering adds
wires on its side of every box.
Rules out: a drawing of `f >> g` that relays `f` out when `g` is added. -/
theorem functor [DecidableEq S.Ob] (X Z : List S.Ob) (b : S.Box)
    {f g : Diagram S} (h : f.cod = g.dom) :
    (Diagram.id X).drawing = []
    ∧ (Diagram.box b).drawing = [⟨0, (S.dom b).length, (S.cod b).length, 0⟩]
    ∧ (Diagram.comp f g).drawing = f.drawing ++ g.drawing
    ∧ (Diagram.whiskerLeft Z f).drawing = f.drawing.map (Strip.shiftLeft Z.length)
    ∧ (Diagram.whiskerRight f Z).drawing
      = f.drawing.map (Strip.shiftRight · Z.length) :=
  ⟨Diagram.drawing_id X, Diagram.drawing_box b, Diagram.drawing_comp h,
    Diagram.drawing_whiskerLeft Z f, Diagram.drawing_whiskerRight f Z⟩

end Statements.Drawing
