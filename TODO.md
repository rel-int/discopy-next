fix https://github.com/rel-int/discopy-next/issues/632 (a next-repo bug; STYLE.md's `eval(repr(x)) == x` fixes what is right). Branch `claude/632-repr-nan-inf`, worktree /home/user/wt/632.

- [x] `matrix.array2string` spells `nan`, `inf`, `-inf` and complex parts so `eval` reads them back
- [x] `tensor.Box`, `optyx.core.diagram.Box` and `photonic.Gate` use the same spelling
- [x] tests: `eval(repr(x))` reads back for `Matrix`, `Tensor`, `tensor.Box` and optyx boxes
- [x] CHANGELOG.md bullet
