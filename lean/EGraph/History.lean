import EGraph.Rebuild

/-!
# Histories

The model of `discopy.egraph.History`: an e-graph and operations on it, what
a search growing an e-graph does to it.
-/

namespace EGraph

/-- An operation on an e-graph, the name of a method and its arguments. -/
inductive Operation where
  | add (head : Nat) (det : Bool) (arity : Nat) (inputs : List Nat)
  | merge (spiders : List Nat)
  | rebuild
deriving DecidableEq, Repr

namespace Graph

/-- The e-graph after an operation. -/
def apply (g : Graph) : Operation → Graph
  | .add head det arity inputs => (g.add head det arity inputs).1
  | .merge spiders => g.merge spiders
  | .rebuild => g.rebuild

/-- The e-graph after each operation in turn, `History.fold`. -/
def fold (g : Graph) (operations : List Operation) : Graph :=
  operations.foldl apply g

end Graph

end EGraph
