toumix, live in Daylight, 11-06, on S6-5 (#445, the two threads on `discopy/group.py:248` here):

> Freeze Action (Recommended): `Action` becomes a frozen dataclass, like the other value classes in `group`, so sharing it is safe; keep caching every call.

- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-06 09:58 `group.Action` is a frozen dataclass, so the one action each group caches cannot be changed under its other callers
