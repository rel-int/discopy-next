import Lean.Data.Json
import Premonoidal.Functor

/-!
The model of diagrams as an oracle: a case names an operation and its
arguments, `{"op": "then", "f": ..., "g": ...}`, or a functor into diagrams
by its images of wires and boxes, `{"op": "functor", "ob": ..., "ar": ..., "f": ...}`, a diagram is written as
`Diagram.encode` writes it, `{"dom": ["x"], "boxes": [{"name": "f",
"dom": ["x"], "cod": ["y"], "offset": 0}]}`, and so is the answer, with its
codomain.
-/

namespace Premonoidal
open Lean

/-- Wires are names and a box is a name, a domain and a codomain. -/
abbrev Named : Signature :=
  ⟨String, String × List String × List String, (·.2.1), (·.2.2)⟩

instance : DecidableEq Named.Ob := inferInstanceAs (DecidableEq String)

instance decidableComposable {S : Signature} [DecidableEq S.Ob] :
    (X : List S.Ob) → (ls : List (Layer S)) → Decidable (Composable X ls)
  | _, [] => isTrue trivial
  | _, l :: ls => @instDecidableAnd _ _ _ (decidableComposable l.cod ls)

/-- `Diagram.decode`: each box goes at its offset on the codomain so far. -/
def decode (dom : List String) (boxes : List (Named.Box × Nat)) :
    Except String (Diagram Named) :=
  let step (so_far : List String × List (Layer Named)) (next : Named.Box × Nat) :=
    let layer : Layer Named := ⟨so_far.1.take next.2, next.1,
      so_far.1.drop (next.2 + (Named.dom next.1).length)⟩
    (layer.cod, so_far.2 ++ [layer])
  let layers := (boxes.foldl step (dom, [])).2
  if h : Composable dom layers then .ok ⟨dom, layers, h⟩
  else .error "the boxes do not compose"

def parse (json : Json) : Except String (Diagram Named) := do
  let dom ← json.getObjValAs? (List String) "dom"
  let boxes ← (← json.getObjValAs? (List Json) "boxes").mapM fun box => do
    return ((← box.getObjValAs? String "name",
      ← box.getObjValAs? (List String) "dom",
      ← box.getObjValAs? (List String) "cod"),
      ← box.getObjValAs? Nat "offset")
  let diagram ← decode dom boxes
  if let .ok cod := json.getObjValAs? (List String) "cod" then
    if diagram.cod ≠ cod then throw "the codomain is not the one given"
  return diagram

/-- `Diagram.encode`, with the codomain. -/
def encode (f : Diagram Named) : Json :=
  Json.mkObj [("dom", toJson (f.dom : List String)), ("cod", toJson (f.cod : List String)),
    ("boxes", toJson <| f.layers.map fun layer => Json.mkObj [
      ("name", toJson layer.box.1), ("dom", toJson layer.box.2.1),
      ("cod", toJson layer.box.2.2), ("offset", toJson layer.left.length)])]

def answer (op : String) (case : Json) : Except String Json := do
  let f ← parse (← case.getObjVal? "f")
  match op with
  | "then" =>
    let g ← parse (← case.getObjVal? "g")
    if f.cod = g.dom then return encode (f.comp g)
    throw "the diagrams do not compose"
  | "tensor" => return encode (f.tensor (← parse (← case.getObjVal? "g")))
  | "whisker_left" =>
    return encode (Diagram.whiskerLeft (← case.getObjValAs? (List String) "ob") f)
  | "whisker_right" =>
    return encode (f.whiskerRight (← case.getObjValAs? (List String) "ob"))
  | "functor" =>
    let ob ← (← case.getObjValAs? (List Json) "ob").mapM fun wire => do
      return (← wire.getObjValAs? String "name", ← wire.getObjValAs? (List String) "ob")
    let ar ← (← case.getObjValAs? (List Json) "ar").mapM fun box => do
      return ((← box.getObjValAs? String "name",
        ← box.getObjValAs? (List String) "dom",
        ← box.getObjValAs? (List String) "cod"), ← parse (← box.getObjVal? "ar"))
    let image (X : List String) : Except String (List String) :=
      X.flatMapM fun x => (ob.lookup x).elim (throw s!"no image for the wire {x}") pure
    discard <| image f.dom
    for layer in f.layers do
      let some g := ar.lookup layer.box | throw s!"no image for the box {layer.box.1}"
      if g.dom ≠ (← image layer.box.2.1) ∨ g.cod ≠ (← image layer.box.2.2) then
        throw s!"the image of the box {layer.box.1} has the wrong domain or codomain"
      discard <| image (layer.left ++ layer.right)
    return encode <| Lift.map (Diagram.premonoidal Named)
      (fun x => (ob.lookup x).getD [x])
      (fun box => (ar.lookup box).getD (Diagram.box box)) f
  | _ => throw s!"unknown operation {op}"

end Premonoidal
