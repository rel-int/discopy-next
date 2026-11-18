> Review round on #623 (cubic-dev-ai[bot]), head f7e57021:
>
> 1. discopy/drawing/drawing.py:341 (thread 4608202627)
> [P1] This representation loses the `Wire` payloads stored in drawing nodes because it relies on `Node.__repr__`, which prints values with `str`; colored or daggered drawings therefore cannot round-trip through `eval(repr(drawing))`. Preserve those payloads in the node representation before claiming evaluable drawing reprs.

- [ ] cubic's P1: whether a coloured or daggered drawing reads back — measure it, pin what holds in the test
