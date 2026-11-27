Every dual rail diagram serialises: `balanced.DualRailBraid`,
`balanced.DualRailTwist`, `ribbon.DualRailCup`, `ribbon.DualRailCap` and
`quantum.zx.Scalar` read back the arguments their own constructors take,
where the inherited `from_tree` passed them a `name` they refuse, so
`loads(dumps(x))` raised a `TypeError` on the image of either `DualRail`
functor and on any diagram holding a ZX scalar ([next#496](https://github.com/rel-int/discopy-next/pull/496)).
