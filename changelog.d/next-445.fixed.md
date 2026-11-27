`quantum.tk.to_tk` drops destructively measured qubits when
`Measure(override_bits=True)` overrides existing bits, so gates and
measurements after it act on the remaining qubits; `QuantumGate` keeps 0-d
phases and backend tensor entries without calling `numpy.ravel`,
`circuit.Box.array` stacks backend scalars with `matrix.asarray`,
`QuantumGate.is_deterministic` detaches tensors before converting to NumPy
and `U1.array` converts its constants in the active backend, so `CRz`,
`CRx`, `CU1` and `QuantumGate` keep gradients under PyTorch and JAX
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
