> Implement the washout period of rel-int/discopy-next#216 in `discopy/quantum/reservoir.py`: `fit(..., washout=0)` drops the first `washout` time steps (rows) of the reservoir's features before the readout is fitted, as the issue specifies. Virtual nodes and the input-qubit measurement stay for later.

- [x] `Reservoir.fit(..., washout=0)` drops the first `washout` rows of features and targets, validating `0 <= washout < len(sequence)`
- [x] Doctest, and a unit test in `test/quantum/reservoir.py` that fails on main
- [ ] CHANGELOG `Added` bullet citing next#216
