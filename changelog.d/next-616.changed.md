`Matrix.__init__` no longer copies an array that already has its dtype:
on numpy the matrix keeps a read-only view of it, so writing into
`m.array` raises and the caller's array stays writeable, and a torch
tensor stays shared with the caller ([next#616](https://github.com/rel-int/discopy-next/pull/616)).
