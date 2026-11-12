# The Lean model of `discopy.egraph`

A pure Lean 4 model of the algorithms of [`discopy/egraph.py`](../discopy/egraph.py),
`partition` so far, with theorems proved on the model. Nothing here is a proof about the Python:
the two are linked by a differential test, the model run as an oracle against
the Python on generated cases.

The model mirrors `discopy/egraph.py` as of `45f9200`. A change to the
algorithms there is followed here, or the differential test says so.

| file | what it holds |
| --- | --- |
| [`Statements.lean`](Statements.lean) | the definitions and the main theorems restated, the one file to read |
| [`EGraph/Partition.lean`](EGraph/Partition.lean) | `partition` and the proof of `partition_least` |
| [`Audit.lean`](Audit.lean) | fails on `sorry`, `native_decide` or an axiom beyond `propext`, `Quot.sound`, `Classical.choice` |
| [`Oracle.lean`](Oracle.lean) | the `oracle` executable: one JSON case per line in, the model's answer out |
| [`test_oracle.py`](test_oracle.py) | the differential test against the Python |

What is modelled, and what is not:

- `partition n pairs` is modelled by its result, the least spider related to
  each spider, not by its union-find: the model rewrites the whole list of
  classes for each pair.
- Spiders are natural numbers and have no type: `partition` takes any pairs,
  in the Python as in the model, and it is `EGraph.__init__` that refuses
  classes relating spiders of two types.

How to build and run it is in
[CONTRIBUTING.md](../CONTRIBUTING.md#run-the-lean-proofs).
