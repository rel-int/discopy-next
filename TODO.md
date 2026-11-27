> Alexis's review of #768, three points:
>
> 1. Move the contents of `dataflow/__init__.py` into `dataflow/symmetric.py`, so the base level holds the mixins. `__init__` keeps only the package docstring, and imports if the package needs any.
>
> 2. Add a `dataflow` subclass of `Hypergraph` with a `followed_by` method (head and tail too, if `to_hypergraph` needs them on arrows), and make it the hypergraph factory of the dataflow diagrams. With that, `to_hypergraph` and `Equation` map into a codomain that has the primitives, so the functor can be strict as originally agreed: it raises when the codomain lacks one, whether the map is callable or a `Mapping`. Drop the Mapping-only deviation if that works. If something still blocks it, say what.
>
> 3. Add axioms for dataflow on the abc pattern, a `DataflowCategory` or whatever name fits the repo, each checked by the property matrix: head and tail on types and arrows as monoidal endofunctors, stated as functor equations; apart from functoriality `x.head.head == x.head`, `x.head.tail == Ty()`, `x.d.head == Ty()`, `x.d.tail == x`, `x.tail == x` for constant `x`; head and tail commute with dagger; `followed_by(x)` is an isomorphism with its dagger as inverse; `followed_by` is natural, `f.head @ f.tail.d >> followed_by(y) == followed_by(x) >> f`, on the nose where it can be and otherwise up to the functor into `stream.Stream`; `followed_by(x.d)` is the identity on `x.d`, up to the unit. No law about `x.tail.head`.

- [x] Move the mixins of dataflow/__init__.py into dataflow/symmetric.py
- [x] Add a dataflow Hypergraph with the primitives and make the functor strict
- [x] Add abc.DataflowCategory with its axioms, checked by the property matrix
