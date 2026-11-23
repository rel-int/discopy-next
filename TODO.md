> taken, no ask: [#696](https://github.com/rel-int/discopy-next/issues/696) is ours and chooses nothing — `Serialisable` states the roundtrip as a law and `path.Matrix` breaks it. Verbatim from the report:
>
> > **What it would take.** `to_tree` adds the four fields and `from_tree` reads them, with `dom` and `cod` the mode counts rather than the array's shape — about the size of `__repr__`, which already names all four. The `repr` roundtrip is tested (`test_repr_is_transparent` in `test/optyx/test_path.py`); a `dumps`/`loads` cell beside it would have caught this.
>
> *Bugs in a next repo*: no public name beyond the protocol it completes, no stated convention changed, no pick between readings. Taken without asking.

- [x] path.Matrix.to_tree writes creations, selections, normalisation and scalar; from_tree reads them back
- [x] test a dumps/loads cell beside test_repr_is_transparent, the report's own matrix among its cases
- [x] CHANGELOG bullet with the PR number, and the gates of next-checks
