> Fix three no-decision follow-ups listed on rel-int/discopy-next#445: `Matrix.lambdify` passes `modules=` twice to sympy's lambdify; `test/optyx/test_zx.py` redefines `H` (flake8 F811); `discopy/optyx/channel.py`'s fallback `lambdify` returns `self`, so `subs` never reaches a box's symbolic data.

- [WIP] @session_01MEJxy5Y19SnBwRFf22vqRj-2026-11-17 19:04 `Matrix.lambdify`: a `modules` keyword replaces the backend's, with a test
- [WIP] @session_01MEJxy5Y19SnBwRFf22vqRj-2026-11-17 19:05 `test/optyx/test_zx.py`: drop the shadowed import of `H`
- [ ] `optyx.channel.Channel.lambdify`: substitute into `kraus` if a symbolic box reaches it, with a test
- [ ] CHANGELOG.md bullet
