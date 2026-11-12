import Lean
import Statements

/-!
Fail unless every theorem of `Statements.lean` rests on nothing but `propext`,
`Quot.sound` and `Classical.choice`: `sorry` is the axiom `sorryAx` and each
`native_decide` an axiom of its own. Run it with `lake env lean Audit.lean`.
-/

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let some module := env.getModuleIdx? `Statements
    | throwError "Statements is not imported"
  let allowed := [``propext, ``Quot.sound, ``Classical.choice]
  let mut audited := #[]
  for (name, info) in env.constants.map₁.toList do
    if env.getModuleIdxFor? name == some module && info matches .thmInfo _ then
      for used in ← collectAxioms name do
        unless allowed.contains used do
          throwError "{name} depends on the axiom {used}"
      audited := audited.push name
  if audited.isEmpty then throwError "Statements has no theorem"
  logInfo m!"{audited.size} theorems audited: {audited.qsort Name.lt}"
