> Fix three no-decision follow-ups listed on rel-int/discopy-next#445: `Matrix.lambdify` passes `modules=` twice to sympy's lambdify; `test/optyx/test_zx.py` redefines `H` (flake8 F811); `discopy/optyx/channel.py`'s fallback `lambdify` returns `self`, so `subs` never reaches a box's symbolic data.

- [x] `Matrix.lambdify`: a `modules` keyword replaces the backend's, with a test
- [x] `test/optyx/test_zx.py`: drop the shadowed import of `H`
- [x] `optyx.channel.Channel.lambdify`: substitute into `kraus` if a symbolic box reaches it, with a test
- [x] CHANGELOG.md bullet
