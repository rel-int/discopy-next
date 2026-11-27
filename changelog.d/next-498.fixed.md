`optyx.core.zx.H` is its own dagger: a box declares itself self-adjoint
with `is_dagger=None`, as `quantum.gates.SWAP` does, so `H[::-1] == H`
and `to_pyzx` translates a daggered Hadamard, which `from_pyzx` reads
back as `H` ([next#498](https://github.com/rel-int/discopy-next/pull/498)).
