> Implement rel-int/discopy-next#8 as USER (toumix) ruled on decision 13 of #56, live 11-09 ('the five as recommended. #8 (option 2, `np.asarray`)'): `Matrix.__init__` uses `np.asarray` instead of copying, and marks the array read-only where the backend allows.

- [WIP] @session_01MEJxy5Y19SnBwRFf22vqRj-2026-11-17 18:44 `Matrix.__init__`: `asarray`, a read-only view on numpy, the sharing documented
- [WIP] @session_01MEJxy5Y19SnBwRFf22vqRj-2026-11-17 18:44 every site writing into a matrix or tensor array after construction builds its array first
- [ ] test: shares memory, read-only, the caller's array writeable
- [ ] CHANGELOG.md bullet
