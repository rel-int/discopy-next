> Review round on #698 (cubic-dev-ai[bot]), head 44eb50eb:
>
> 1. discopy/optyx/core/path.py:389 (thread 4661073130)
> [P2] `Matrix.from_tree` now rejects every pre-change dump of a profile-less path matrix with `KeyError: 'creations'`. Read the new fields with constructor defaults so existing serialized matrices remain loadable.
>
> ```suggestion
>             tuple(tree.get('creations', ())),
>             tuple(tree.get('selections', ())),
>             tree.get('normalisation', 1),
>             tree.get('scalar', 1))
> ```

- [ ] discopy/optyx/core/path.py:389 `Matrix.from_tree` now rejects every pre-change dump of a profile-less path matrix with `KeyError: 'creations…
