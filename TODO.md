> Fix quantum bugs from #445: to_tk destructive override_bits qubit tracking and QuantumGate torch requires_grad

- [x] Drop destructively measured qubits in to_tk when Measure(override_bits=True)
- [WIP] @agy-445-2026-11-26 12:43 Handle 0-d and requires_grad backend tensors in QuantumGate.__init__ without numpy.ravel, plus B904 in pennylane.py
- [ ] Add CHANGELOG.md entry
