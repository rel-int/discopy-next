`eval(repr(x)) == x`, `STYLE.md`'s transparency law, holds for the fifteen box
classes that broke it in the two copies of `zx` and in `optyx.core`. Each
printed the generic box call — name, dom, cod and a `data=` keyword — which
their own narrowed constructors reject: `zw.Endo`, `zw.Add`, `zw.Multiply`,
`zw.Divide`, `zw.Mod2`, `zx.And`, `zx.Or`, `diagram.Scalar`,
`diagram.DualRail`, `diagram.PhotonThresholdDetector` and
`quantum.zx.Scalar` raised `TypeError`, `zw.W` and
`diagram.EmbeddingTensor` raised `NameError` on an unqualified name, and
`zw.Create` and `zw.Select` read back silently unequal, their photons being
their own name, dom and cod. Each now prints the call to its public
constructor and keeps its data where `==` reads it — `zw.W(2,
is_dagger=True)`, `diagram.EmbeddingTensor(1, 2)`, `quantum.zx.Scalar(2)` —
the shape [next#445](https://github.com/rel-int/discopy-next/issues/445) ruled for `quantum.gates.Parametrized` and
`optyx.qubits.BitControlledGate`. `diagram.EmbeddingTensor` keeps its
`input_dim` and `output_dim` as public attributes to print them, and the
`listed` helper for internal states moves from `optyx.photonic` to
`optyx.utils.misc`, where the core can reach it. `test/optyx/test_repr.py`
now quantifies the law over the core as well as the tower, and
`test/quantum/zx.py` over the ZX boxes, `Swap` included. `zw.Endo` also
reads its scalar rather than the name derived from it in `setoid`, so that two
endomorphisms with equal scalars are equal: `Endo(0.5).conjugate() ==
Endo(0.5)` was false, the conjugate's name printing the negative zero
`(0.5-0j)` that the scalar does not have, and no representation of it could
have read back ([next#472](https://github.com/rel-int/discopy-next/issues/472), [next#468](https://github.com/rel-int/discopy-next/issues/468), [next#168](https://github.com/rel-int/discopy-next/issues/168)).
