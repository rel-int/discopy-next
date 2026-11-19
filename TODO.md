> From [#636](https://github.com/rel-int/discopy-next/issues/636), filed by `toumix-agents`
> 2026-11-18 16:29Z, whose own last line sets the default rather than asking for a word:
>
> > `monoidal.Ty.id(Colour('black'))` prints `monoidal.Ty.id(monoidal.Colour('black'))` and reads
> > back. The `rigid` one, and so `pivotal` to `frobenius`, does not. The adjoint of the empty type of
> > a colour should be the empty type of that colour, since `z` swaps colours and both sides are the
> > same colour.
> >
> > Default: an agent fixes `rigid.Ty`'s adjoint and repr for the empty coloured type in its own PR.
> > Asked 2026-11-18.

- [x] rigid.Ty.l and .r keep the colour of the empty type, so t.r.l == t
- [x] The empty coloured type's repr reads back in biclosed and below, not only in monoidal
- [x] Pin both down the hierarchy — rigid, pivotal, frobenius — and the changelog bullet
