> Implement the ruling of rel-int/discopy-next#560 in rel-int/discopy-next: a Box is a Bubble with zero args, and `Bubble` goes. Read the issue and all its comments first; toumix (the owner) ruled on its open points live on 2026-11-25:
>
> - keep an alias: `Bubble` stays importable for now as an alias of the fused class (a `Box` with `args`), rather than disappearing outright
> - `Sum` doesn't fuse yet: leave `discopy.sum` / `Sum` as it is
> - `tensor.Bubble`'s `func` (the function it applies elementwise, compared in its `==` and hash, see #506) moves to a subclass of `tensor.Box` called `Elementwise`; a plain tensor `Box` carries no `func`
>
> In maths terms: a box is an operation symbol of arity n, taking n arrows `args` to an arrow dom → cod, a generator being n = 0; a functor sends a box with args by `getattr(cod, method)(*map(self, args), dom=..., cod=...)`. `cat.Bubble(Box)` and the per-module `Bubble` classes plus each `bubble_factory` are what fuse; `isinstance(f, Bubble)` checks become `f.args`.
>
> Follow AGENTS.md, RULES.md (TODO.md with the prompt verbatim, checkbox mutex via the `checkboxes` skill), STYLE.md and CONTRIBUTING.md, and run the `next-checks` skill's gates before pushing. Keep it to one reviewable pull request if you can; if it is far past a thousand lines, propose a slicing on the PR instead of pushing it all at once.
>
> Heads-up on a collision: draft PR rel-int/discopy-next#728 (branch `amp/left-to-right-drawing`, from thread T-48aec84b-09cc-47fa-bf08-e9067aece821) changes `monoidal.Bubble.__init__`/`to_drawing`, `drawing/backend.py` and `config.BOX_DRAWING_ATTRIBUTES` to draw a bubble's args left to right. Whichever lands second merges the other in; don't edit that branch.
>
> Do the work yourself and don't create other threads. When you open the PR, send a short note with its link to thread T-48aec84b-09cc-47fa-bf08-e9067aece821 with send_thread_message.

- [x] cat: `Box` takes `args` and `method`, `Arrow.bubble` builds a box with args, the functor sends one by `getattr(cod, method)`, `Bubble` an alias
- [WIP] @T-14209019-2026-11-25 14:31 every module's `Bubble` fuses into its `Box` and `bubble_factory` goes; `isinstance(f, Bubble)` becomes `f.args`
- [WIP] @T-14209019-2026-11-25 14:31 tensor: `Elementwise(Box)` carries `func`, a plain tensor `Box` none; the same in `quantum.circuit` and `quantum.zx`
- [ ] tests, notebooks and ADRs follow; CHANGELOG bullet
- [ ] run next-checks
