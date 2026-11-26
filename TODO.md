> Fix quantum bugs from #445: to_tk destructive override_bits qubit tracking and QuantumGate torch requires_grad

- [x] Drop destructively measured qubits in to_tk when Measure(override_bits=True)
- [x] Handle 0-d and requires_grad backend tensors in QuantumGate.__init__ without numpy.ravel, plus B904 in pennylane.py
- [x] Add CHANGELOG.md entry
