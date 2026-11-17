> Fix rel-int/discopy-next#606: `neural.train.per_board_backward`'s doctest fails on some random draws, because it builds `torch.nn.Linear(2, 1)` and `torch.randn(...)` with no seed and compares in float32 with `torch.allclose` at its default tolerances. Seed the example with `torch.manual_seed(0)`, following the pattern `discopy/neural/act.py` already uses (`>>> _ = torch.manual_seed(0)` with the `# doctest: +EXTRA` marker convention where it applies).

- [x] Seed the `per_board_backward` example with `torch.manual_seed(0)`
- [ ] Show the doctest passes whatever the prior random state, 20 runs of `pytest --doctest-modules discopy/neural/train.py -n 4`
