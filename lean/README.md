# The Lean model of `discopy.egraph`

A pure Lean 4 model of the algorithms of [`discopy/egraph.py`](../discopy/egraph.py),
`partition`, the pure `EGraph` and the `Builder`, with theorems proved on the
model. Nothing here is a proof about the Python:
the two are linked by a differential test, the model run as an oracle against
the Python on generated cases.

The model mirrors `discopy/egraph.py` as of `45f9200`. A change to the
algorithms there is followed here, or the differential test says so.

| file | what it holds |
| --- | --- |
| [`Statements.lean`](Statements.lean) | the definitions and the main theorems restated, the one file to read |
| [`EGraph/Partition.lean`](EGraph/Partition.lean) | `partition` and the proof of `partition_least` |
| [`EGraph/Classes.lean`](EGraph/Classes.lean) | lists of classes: normal ones are equal when they relate the same spiders |
| [`EGraph/Graph.lean`](EGraph/Graph.lean) | the e-graph, `merge` and `add`, with the lemmas `merge_equivalence` is made of and `add_hashcons` |
| [`EGraph/Rebuild.lean`](EGraph/Rebuild.lean) | `rebuild`, its termination, and that it is the least congruent equivalence |
| [`EGraph/History.lean`](EGraph/History.lean) | operations and the fold of a history |
| [`EGraph/UnionFind.lean`](EGraph/UnionFind.lean) | the labels of the builder: `root` with path halving, which ends on the root and keeps every root |
| [`EGraph/Builder.lean`](EGraph/Builder.lean) | the `Builder`, method by method, and `build`, what it freezes into after a history |
| [`EGraph/Invariant.lean`](EGraph/Invariant.lean) | the invariant of the builder, stated on the root of each spider |
| [`EGraph/Hashed.lean`](EGraph/Hashed.lean) | entering a node in the hash-cons keeps it |
| [`EGraph/Place.lean`](EGraph/Place.lean) | placing a node keeps it |
| [`EGraph/Link.lean`](EGraph/Link.lean) | a class joining another keeps it |
| [`EGraph/Union.lean`](EGraph/Union.lean) | `union` keeps it, relates its two spiders and lowers what `rebuild` has left to do |
| [`EGraph/Drain.lean`](EGraph/Drain.lean) | `rebuild` keeps it and ends with an empty worklist |
| [`EGraph/Freeze.lean`](EGraph/Freeze.lean) | what a builder freezes into, where `merge` and `rebuild` are those of the e-graph |
| [`EGraph/Add.lean`](EGraph/Add.lean) | appending a node on new spiders keeps the invariant |
| [`EGraph/Building.lean`](EGraph/Building.lean) | `add` is that of the e-graph, the builder of an e-graph, and the laws `building` and `adding` |
| [`Statements/Premonoidal.lean`](Statements/Premonoidal.lean) | the same for diagrams: strict premonoidal categories, Diagram(S) and what is proved of it |
| [`Premonoidal/Basic.lean`](Premonoidal/Basic.lean) | categories and strict premonoidal categories |
| [`Premonoidal/Diagram.lean`](Premonoidal/Diagram.lean) | Diagram(S), the proof that it is strict premonoidal and not monoidal |
| [`Premonoidal/Functor.lean`](Premonoidal/Functor.lean) | strict premonoidal functors, the functor out of Diagram(S) and its uniqueness |
| [`Premonoidal/Oracle.lean`](Premonoidal/Oracle.lean) | diagrams read and written as `Diagram.encode` does, for the oracle |
| [`test_premonoidal.py`](test_premonoidal.py) | the differential test of diagrams against `monoidal.Diagram` |
| [`Audit.lean`](Audit.lean) | fails on `sorry`, `native_decide` or an axiom beyond `propext`, `Quot.sound`, `Classical.choice` |
| [`Oracle.lean`](Oracle.lean) | the `oracle` executable: one JSON case per line in, the model's answer out |
| [`test_oracle.py`](test_oracle.py) | the differential test against the Python |

What is modelled, and what is not:

- `partition n pairs` is modelled by its result, the least spider related to
  each spider, not by its union-find: the model rewrites the whole list of
  classes for each pair.
- An e-graph is a list of nodes and the class of each spider: a node is a
  box, given by a number, whether it is deterministic, and its input and
  output spiders. There is no `Hypergraph`: no domain or codomain, and
  left-monogamy and acyclicity are neither modelled nor needed.
- `rebuild` runs on fuel, the sum of the classes, where the Python loops
  until nothing changes: `rebuild_congruent` shows the fuel is enough.
- `add` leaves out the `partition` of `EGraph.__init__`, which changes nothing
  on normal classes.
- The `Builder` keeps `least` and `parents` in arrays where the Python has
  dictionaries, has no `heads` or `known`, since a node carries the number of
  its box, and cuts `union` in three methods, `union`, `join` and `link`. Its
  loops, in `root` and `rebuild`, run on fuel, which `building` shows is
  enough. That law, a builder freezes into what its operations give in turn,
  is proved for the histories the Python accepts: boxes on spiders that are
  there, with as many outputs as the nodes of their head.
- Spiders are natural numbers and have no type: `partition` takes any pairs,
  in the Python as in the model, and it is `EGraph.__init__` that refuses
  classes relating spiders of two types. A valid e-graph of the model has
  one number of outputs for each head, which the Python reads off the type.

## Diagrams are the morphisms of a premonoidal category

[`Statements/Premonoidal.lean`](Statements/Premonoidal.lean) models
[`monoidal.Diagram`](../discopy/monoidal.py) as `Diagram.encode` writes it, a
domain and a list of boxes with offsets, and proves that these diagrams are a
strict premonoidal category in which interchange holds for two boxes only
when they are one and the same scalar: with any other box, Diagram(S) is not
monoidal. It is the free one: generating objects and boxes sent into a strict
premonoidal category extend to exactly one strict premonoidal functor. The
model mirrors
`discopy/monoidal.py` as of `c842a24`.

- Morphisms are untyped, with `dom` and `cod`, as in the Python, so that
  `(X ⊗ Y) ⋊ f = X ⋊ (Y ⋊ f)` is an equality of morphisms with no transport
  along `(X ++ Y) ++ Z = X ++ (Y ++ Z)`. Composition is total and every law
  about it assumes its arguments composable: where the Python raises, the
  model returns its first argument and nothing is said of it.
- A functor is tested against `monoidal.Functor` into diagrams only, where
  it sends each box to a diagram; the theorem is about any codomain.
- One box per layer: a `Layer` with several boxes, which `foliation` makes
  and `encode` flattens, is not modelled.
- No colours: types are lists of wires, the free monoid, where a coloured
  `Ty` is a path and its tensor is partial.
- `Sum`, `Bubble` and the subclasses of `Diagram` are not modelled.

How to build and run it is in
[CONTRIBUTING.md](../CONTRIBUTING.md#run-the-lean-proofs).
