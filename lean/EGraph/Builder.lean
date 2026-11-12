import Std.Data.HashMap
import EGraph.UnionFind
import EGraph.History

/-!
# The builder

The model of `discopy.egraph.Builder`, method by method: a union-find over
the spiders, the least spider of each class, a key for each node, the
hash-cons from a key to its first node, the parents of each class and the
worklist of pairs of nodes found congruent. Each method takes the builder
and gives it back, where the Python changes it in place.

Two dictionaries of the Python are arrays here: `least[root]` is the root
where the Python has no entry, and `parents[root]` is empty where it has
none. `heads` and `known` number the boxes, which a node has done already.
`Builder.union` is three methods here, `union`, `join` and `link`: finding
the two roots, choosing which joins the other, and moving its parents. The
loops of `root` and `rebuild` run on fuel.
-/

namespace EGraph

structure Builder where
  label : Array Nat
  least : Array Nat
  nodes : Array Node
  keys : Array (List Nat)
  hashcons : Std.HashMap (List Nat) Nat
  parents : Array (List Nat)
  worklist : List (Nat × Nat)

namespace Builder

/-- The root of the class of a spider, halving the path there. -/
def root (b : Builder) (spider : Nat) : Builder × Nat :=
  let found := EGraph.root b.label b.label.size spider
  ({ b with label := found.1 }, found.2)

/-- The roots of spiders in turn, `map(self.root, spiders)`. -/
def roots (b : Builder) : List Nat → Builder × List Nat
  | [] => (b, [])
  | spider :: spiders =>
    let first := b.root spider
    let rest := first.1.roots spiders
    (rest.1, first.2 :: rest.2)

/-- The class of a spider, the least spider related to it. -/
def find (b : Builder) (spider : Nat) : Builder × Nat :=
  let found := b.root spider
  (found.1, found.1.least.getD found.2 found.2)

/-- Enter a node in the hash-cons under its key: the least node of a key
stays, any other is congruent to it, their pair put on the worklist. -/
def cons (b : Builder) (box : Nat) : Builder :=
  let key := b.keys.getD box []
  match b.hashcons[key]? with
  | none => { b with hashcons := b.hashcons.insert key box }
  | some first =>
    if first = box then b
    else { b with
      hashcons := b.hashcons.insert key (min first box)
      worklist := b.worklist ++ [(first, box)] }

/-- Append a node with its key: hash-consed and a parent of each root of its
key when it is deterministic, nothing for a sample, whose key is empty. -/
def place (b : Builder) (node : Node) (key : List Nat) : Builder :=
  let box := b.nodes.size
  let b := { b with nodes := b.nodes.push node, keys := b.keys.push key }
  if key = [] then b
  else key.tail.foldl (fun b root =>
    { b with parents := b.parents.modify root (· ++ [box]) }) (b.cons box)

/-- Add a box reading some spiders and give its outputs: the next spiders,
or those of the first deterministic node of its head on their classes. -/
def add (b : Builder) (head : Nat) (det : Bool) (arity : Nat)
    (inputs : List Nat) : Builder × List Nat :=
  let rooted := if det then b.roots inputs else (b, [])
  let key := if det then head :: rooted.2 else []
  let b := rooted.1
  match (if det then b.hashcons[key]? else none) with
  | some first => (b, ((b.nodes[first]?).map (·.outputs)).getD [])
  | none =>
    let outputs := List.range' b.label.size arity
    let b := { b with
      label := b.label ++ outputs.toArray
      least := b.least ++ outputs.toArray
      parents := b.parents ++ Array.replicate arity [] }
    (b.place ⟨head, det, inputs, outputs⟩ key, outputs)

/-- Move a node to its new key in the hash-cons. -/
def rekey (b : Builder) (box : Nat) : Builder :=
  match b.keys.getD box [] with
  | [] => b
  | head :: inputs =>
    let rooted := b.roots inputs
    { rooted.1 with keys := rooted.1.keys.setIfInBounds box (head :: rooted.2) }.cons box

/-- The class of the root `right` joins that of the root `left`: each of its
parents leaves the hash-cons, then moves there to its new key, where it may
meet a node now congruent to it. -/
def link (b : Builder) (left right : Nat) : Builder :=
  let moved := b.parents.getD right []
  let least := b.least.getD right right
  let b := { b with
    parents := b.parents.setIfInBounds right []
    hashcons := moved.foldl
      (fun hashcons box => hashcons.erase (b.keys.getD box [])) b.hashcons
    label := b.label.setIfInBounds right left
    least := if least < b.least.getD left left
      then b.least.setIfInBounds left least else b.least }
  let b := moved.foldl rekey b
  { b with parents := b.parents.modify left (· ++ moved) }

/-- Relate two roots: the one with fewer parents joins the other, the greater
if they have as many, at once when it has no parent. -/
def join (b : Builder) (left right : Nat) : Builder :=
  if left = right then b
  else
    let lesser := min left right
    let greater := max left right
    if b.parents.getD greater [] = [] then
      { b with label := b.label.setIfInBounds greater lesser }
    else
      let fewer : Bool :=
        (b.parents.getD lesser []).length < (b.parents.getD greater []).length
      b.link (if fewer then greater else lesser) (if fewer then lesser else greater)

/-- Relate two spiders, egg's union: their roots join. -/
def union (b : Builder) (left right : Nat) : Builder :=
  let first := b.root left
  let second := first.1.root right
  second.1.join first.2 second.2

/-- Relate spiders, without closing under congruence. -/
def merge (b : Builder) (spiders : List Nat) : Builder :=
  (Graph.chain spiders).foldl (fun b pair => b.union pair.1 pair.2) b

/-- The outputs of a node. -/
def outputs (b : Builder) (box : Nat) : List Nat :=
  ((b.nodes[box]?).map (·.outputs)).getD []

/-- Relate the outputs of each pair of nodes on the worklist, at most `fuel`
of them. -/
def drain : Nat → Builder → Builder
  | 0, b => b
  | fuel + 1, b =>
    match b.worklist.getLast? with
    | none => b
    | some pair =>
      drain fuel (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
        (fun b pair => b.union pair.1 pair.2) { b with worklist := b.worklist.dropLast })

/-- How many parents there are in all. -/
def total (b : Builder) : Nat :=
  ((List.range b.label.size).map fun root => (b.parents.getD root []).length).sum

/-- The fuel of `rebuild`: each pair leaves the worklist, and each class
that joins another puts at most as many pairs there as there are parents. -/
def fuel (b : Builder) : Nat :=
  b.worklist.length + b.label.size * (b.total + 1)

/-- The congruence closure: relate the outputs of each pair of nodes found
congruent, until none is left to find. -/
def rebuild (b : Builder) : Builder := drain b.fuel b

/-- The classes of the spiders, `map(self.find, range(len(self.label)))`. -/
def classes (b : Builder) : List Nat :=
  ((List.range b.label.size).foldl (fun (state : Builder × List Nat) spider =>
    let found := state.1.find spider
    (found.1, state.2 ++ [found.2])) (b, [])).2

/-- The e-graph built so far, as it is: not rebuilt. -/
def freeze (b : Builder) : Graph := ⟨b.nodes.toList, b.classes⟩

/-- A builder starting from an e-graph. -/
def start (g : Graph) : Builder :=
  g.nodes.foldl (fun b node =>
      let rooted := if node.det then b.roots node.inputs else (b, [])
      rooted.1.place node (if node.det then node.head :: rooted.2 else []))
    { label := g.classes.toArray, least := (List.range g.classes.length).toArray,
      nodes := #[], keys := #[], hashcons := {},
      parents := Array.replicate g.classes.length [], worklist := [] }

/-- The builder after an operation. -/
def apply (b : Builder) : Operation → Builder
  | .add head det arity inputs => (b.add head det arity inputs).1
  | .merge spiders => b.merge spiders
  | .rebuild => b.rebuild

/-- What a builder freezes into after the operations, `History.build`. -/
def build (g : Graph) (operations : List Operation) : Graph :=
  (operations.foldl apply (start g)).freeze

end Builder

end EGraph
