> cubic-dev-ai review on #688 (https://github.com/rel-int/discopy-next/pull/688#pullrequestreview-6091760213):
>
> > discopy/drawing/optimiser.py:1236 [P2] `width` ignores the box corners added by `framed()`, so a wide `min_width` box can extend beyond the declared drawing width and default canvas. Include the corner x-coordinates when computing `width`.

- [ ] optimiser.py:1236 the width and height of a pivotal drawing hold its box corners
