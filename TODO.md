# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Tonight's item is [#443](https://github.com/rel-int/discopy-next/issues/443), filed and planned
by 🐦 Birdsong this morning: three `PLW0108` sites that `ruff == 0.14.0` cannot see because the
rule is preview there, and that the next bump past 0.14 turns `lint` red on. Two of the three
decide nothing and are taken here; the third,
`python/multiplicative.py:295`, moves a module-level lookup from call time to class-body time and
waits on the 🚀 of
[the comment on #443](https://github.com/rel-int/discopy-next/issues/443#issuecomment-6650397574).

- [x] `discopy/stream.py` `Ty.singleton`: `_later=lambda: cls()` → `_later=cls`
- [x] `discopy/stream.py` `Stream.singleton`: `_later=lambda: cls.id()` → `_later=cls.id`
- [x] `CHANGELOG.md` entry under `Project`
- [x] `ruff check discopy` at 0.16.9 passes, pinned `ruff`, `pflake8`, `pylint` and `ty` still pass
- [x] `pytest test/stream.py test/python/ discopy/stream.py discopy/python/ --skip-extra` green
