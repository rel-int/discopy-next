> taken, no ask: point 3 of [#718](https://github.com/rel-int/discopy-next/issues/718) chooses nothing. Verbatim from the report:
>
> > 3. **`pickling`**: `pickle.dumps(H)` and `pickle.dumps(G.from_diagram(X(1, 1)))` raise `PicklingError`, because `zx.py:649` sets `H.dagger = lambda: H` on the instance. Any hypergraph of an `X` spider contains `H`.
> >
> > 3 is a plain bug (a `dagger` method on a class of its own would fix it).
>
> *Bugs in a next repo*: the fix takes the convention `circuit.Box.dagger` and optyx's `Box.dagger` already state, *a box declares itself self-adjoint with `is_dagger=None`*, so no public name and no pick between readings. Points 1, 2 and 4 of #718 are choices and stay with the issue.

- [ ] zx.H is self-adjoint by `is_dagger=None`, which `zx.Box` reads in `dagger`, `__repr__` and `to_tree`, so it pickles and its repr and tree stay transparent
- [ ] a corpus entry for the `pickling` cells of `zx.Diagram` and `zx.Hypergraph`, a Hadamard and a red spider
- [ ] CHANGELOG bullet with the PR number, and the gates of next-checks
