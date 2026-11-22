> Review round on #694 (cubic-dev-ai[bot]), head 49a0f0aa:
>
> 1. discopy/frobenius.py:121 (thread 4655968220)
> [P2] Legacy pickles bypass `__init__`, so `rigid.Wire.__setstate__` can restore a nonzero `_z` without normalization. Normalize `z` in `frobenius.Wire.__setstate__` as well.
>
> ```suggestion
>     def __setstate__(self, state):
>         super().__setstate__(state)
>         self.z %= len(self.windings)
>
>     def __init__(self, *args, **params):
>         super().__init__(*args, **params)
>         self.z %= len(self.windings)
> ```

- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 01:36 frobenius.Wire.__setstate__ normalises z too, so an unpickled wire has no winding either
