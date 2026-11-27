`optyx.photonic.TBS` and `MZI` are the matrix they say they are once
conjugated: the global phase **XORs** `is_gate_dagger` and `is_conj` where it
ORed them, so a conjugated gate carried the daggered gate's phase and
`g >> g.dagger()` was not the identity; and `decomp` conjugates the beam
splitters and phase shifters it decomposes into, where it dropped `is_conj`
altogether, so the gradient of a conjugate — which is the gradient of the
decomposition — was the gradient ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
