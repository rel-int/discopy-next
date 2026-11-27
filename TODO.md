> Alexis's ruling: the stream primitives leave `discopy.feedback` and get their own module, `discopy.dataflow` (after *Monoidal Streams for Dataflow Programming*, DiLavoreEtAl22; `fby` is Lucid's). `discopy.stream` stays the semantics.
>
> Design agreed:
> - `dataflow.Diagram` is a Markov feedback diagram with the stream structure: `class Diagram(<mixin>, feedback.markov or markov.Diagram)` (pick the right Markov base on the branch). The structure lives in mixins, one for diagrams (`head`, `tail`, `followed_by`, their factories) and one for `Ty`/`Wire` (`HeadOb`, `TailOb`, `is_constant`), so another level later is one class with the mixin, nothing generated. Only Markov now.
> - `dataflow.Functor` maps the primitives into a codomain that has them (today only `stream.Stream`/`stream.Ty`) and raises otherwise.
> - Removed with no alias (ADR 1): `HeadOb`, `TailOb`, `Head`, `Tail`, `FollowedBy`, `Wire.is_constant` from `feedback.py`; their entries in `structural_boxes`/`structural_wires`, so no per-level copies; the `hasattr` head/tail/fby branches of `feedback.Functor.__call__`; the optyx and quantum.circuit heads from next#665 (type checks only, no user). This also ends the two silent cases: `Dim`/`Nat`/`Dims` with `head = tail = self`, and `f.head` mapping to an anonymous bubble.
> - Move the interpreted uses to `dataflow`: the fib and random-walk doctests in `stream.py` (~48-102), `test_walk`/`test_fibonacci` in `test/markov.py` (~237-290), the categorical-deep-learning notebook (~231-257), the `markov.py:176-186` walk. Plumbing tests (~30: serialisation, dagger, delay, generated subclasses, functor boundary) move to `test/dataflow.py` or go.
> - Add `dataflow` to the docs API (syntax section) and a CHANGELOG line; an ADR if it fits the repo's practice.

- [x] discopy/dataflow.py: the mixins, Diagram on feedback.markov, Functor raising into a codomain without streams
- [x] remove HeadOb, TailOb, Head, Tail, FollowedBy, is_constant from feedback and every level, structural_wires, the Functor branches, Dim/Nat/Dims heads, circuit and optyx heads
- [x] move the fib, random-walk and walk doctests, test_walk/test_fibonacci and the notebook to dataflow; plumbing tests to test/dataflow.py
- [x] docs API, CHANGELOG, ADR
- [x] next-checks green, PR opened
- [x] Alexis's layout change: dataflow is a package, mixins in __init__, dataflow.symmetric the base level, dataflow.markov over it and feedback.markov
