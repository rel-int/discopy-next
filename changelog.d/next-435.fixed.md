`docs/notebooks/optyx/photon_distinguishability.md` composes its photonic
circuits from `optyx.photonic`, where it used to take `Create` from
`optyx.core.zw` and `DualRail` from `optyx.core.diagram` and compose those
core boxes with channel-level gates: `Create(1, 1, internal_states=...) >>
BS` raised `TypeError: Expected optyx.core.diagram.Diagram, got
optyx.photonic.BBS`, so the notebook never reached the Hong-Ou-Mandel effect
its prose describes. `photonic`'s own module docstring states the
composition, and `photonic.DualRail(n, internal_states=...)` is the class
whose signature the notebook was already calling. The two cells that do
demonstrate the core boxes keep them, module-qualified ([next#435](https://github.com/rel-int/discopy-next/issues/435)).
