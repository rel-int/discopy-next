> Codex review of 25708c979 (https://github.com/rel-int/discopy-next/pull/686#pullrequestreview-6005462633):
>
> > **P1** Honor dagger flags when accumulating the optical unitary: `(u >> u.dagger()).bang().to_channel()` compiles to the second quantisation of `U @ U` instead of the identity; the direct `Box.to_channel()` implementation also needs to honor this flag.
> >
> > **P2** Preserve the optyx category for terms without constants: `qubit.bang()(lambda x: x)` evaluates to a plain `bang.Diagram`, not an optyx `Term`, so `.to_channel()` raises `AttributeError`.

- [x] a daggered box compiles to the conjugate transpose of its unitary, alone and under a bang
- [x] optyx terms of their own, so a box-free term evaluates to an optyx.Term
