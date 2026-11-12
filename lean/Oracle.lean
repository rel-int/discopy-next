import Lean.Data.Json
import EGraph

/-!
The model as an executable: one JSON case per line on the standard input and
what the model gives for it on the standard output. A case with `pairs` is a
partition, `{"spiders": 4, "pairs": [[3, 1], [1, 2]]}` giving
`{"partition": [0, 1, 1, 1]}`. Any other is a history: `nodes`, each
`[head, det, inputs, outputs]`, the `classes` of the spiders and `operations`,
each `["add", head, det, arity, inputs]`, `["merge", spider, ...]` or
`["rebuild"]`, giving the `nodes` and `classes` of the `fold` and of what
the builder freezes into, the `build`, from the partition of the classes as
`EGraph.__init__` takes it. A case may ask for one of them alone,
`"run": ["build"]`: the fold goes through every node at each step.
-/

open Lean EGraph

def entry (α : Type) [FromJson α] (array : Array Json) (i : Nat) : Except String α :=
  fromJson? (array[i]?.getD Json.null)

def node (json : Json) : Except String Node := do
  let array ← json.getArr?
  return ⟨← entry Nat array 0, ← entry Bool array 1,
    ← entry (List Nat) array 2, ← entry (List Nat) array 3⟩

def operation (json : Json) : Except String Operation := do
  let array ← json.getArr?
  let name ← entry String array 0
  if name == "add" then
    let inputs ← entry (List Nat) array 4
    return .add (← entry Nat array 1) (← entry Bool array 2) (← entry Nat array 3) inputs
  if name == "merge" then
    return .merge (← (array.toList.drop 1).mapM fromJson?)
  if name == "rebuild" then
    return .rebuild
  throw s!"unknown operation {name}"

def graph (g : Graph) : Json :=
  Json.mkObj [("classes", toJson g.classes), ("nodes", Json.arr (g.nodes.toArray.map
    fun n => Json.arr #[toJson n.head, toJson n.det, toJson n.inputs, toJson n.outputs]))]

def answer (line : String) : Except String Json := do
  let case ← Json.parse line
  if let .ok pairs := case.getObjValAs? (List (Nat × Nat)) "pairs" then
    let spiders ← case.getObjValAs? Nat "spiders"
    return Json.mkObj [("partition", toJson (partition spiders pairs))]
  let nodes ← (← case.getObjValAs? (List Json) "nodes").mapM node
  let classes ← case.getObjValAs? (List Nat) "classes"
  let operations ← (← case.getObjValAs? (List Json) "operations").mapM operation
  let classes := partition classes.length (links classes)
  let run := (case.getObjValAs? (List String) "run").toOption.getD ["fold", "build"]
  let fold := if run.contains "fold"
    then [("fold", graph (Graph.fold ⟨nodes, classes⟩ operations))] else []
  let build := if run.contains "build"
    then [("build", graph (Builder.build ⟨nodes, classes⟩ operations))] else []
  return Json.mkObj (fold ++ build)

partial def main : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let rec loop : IO UInt32 := do
    let line ← stdin.getLine
    if line.isEmpty then return 0
    match answer line with
    | .ok json => stdout.putStrLn json.compress; loop
    | .error message => IO.eprintln s!"{message}: {line}"; return 1
  loop
