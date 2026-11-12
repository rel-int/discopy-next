import Lean.Data.Json
import EGraph

/-!
The model as an executable: one JSON case per line on the standard input,
`{"spiders": 4, "pairs": [[3, 1], [1, 2]]}`, and what the model gives for it
on the standard output, `{"partition": [0, 1, 1, 1]}`.
-/

open Lean

def answer (line : String) : Except String Json := do
  let case ← Json.parse line
  let spiders ← case.getObjValAs? Nat "spiders"
  let pairs ← case.getObjValAs? (List (Nat × Nat)) "pairs"
  return Json.mkObj [("partition", toJson (EGraph.partition spiders pairs))]

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
