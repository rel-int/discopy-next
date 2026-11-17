> Implement the washout period of rel-int/discopy-next#216 in `discopy/quantum/reservoir.py`: `fit(..., washout=0)` drops the first `washout` time steps (rows) of the reservoir's features before the readout is fitted, as the issue specifies. Virtual nodes and the input-qubit measurement stay for later.

- [WIP] @session_01MEJxy5Y19SnBwRFf22vqRj-2026-11-17 17:28 `Reservoir.fit(..., washout=0)` drops the first `washout` rows of features and targets, validating `0 <= washout < len(sequence)`
- [ ] Doctest, and a unit test in `test/quantum/reservoir.py` that fails on main
- [ ] CHANGELOG `Added` bullet citing next#216
