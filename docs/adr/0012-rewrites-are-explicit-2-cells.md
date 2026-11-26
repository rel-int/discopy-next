# 12. Rewrites are explicit 2-cells

## Context

[#706](https://github.com/rel-int/discopy-next/pull/706) animated isotopies as gifs:
`monoidal.Equation.path` listed the diagrams between two terms through their normal form, and
`to_gif(tween=n)` moved each drawing to the next by `Drawing.interpolate`, which guessed the move
from two drawings. A list of diagrams forgets the moves: the two interchangers of two scalars have
the same domain and codomain and wind in opposite directions (Eckmann–Hilton), so no guess can
tell them apart. Nothing in the library held a composite of rewrite steps: `monoidal.Match` is a
context with one hole, `testable.Equation` is unoriented and forgets the axiom it instantiates,
and `hypergraph.Rewriter` and `egraph` rewrite without keeping their steps. toumix ruled on the
design in thread
[T-e8a5da2e](https://ampcode.com/threads/T-e8a5da2e-b021-4121-9f01-2fa82f517763) on 2026-11-25.

## Decision

`discopy.rewrite` holds the 2-cells between diagrams. A `Rewrite[category]` is a `cat.Arrow`
whose objects are the diagrams of the `category`, a class of diagrams as for `CMap`, and whose
generators are `Layer`s: a diagram, its context, with a hole for each
`Step` that happens at once. A `Step` is a generating 2-cell; its subclasses are the instances of
the axioms of `discopy.abc` that `monoidal.Diagram` breaks strictly: `Interchange` for
`bifunctoriality` with identities, `Slide` for `braid_naturality` with an identity. Each level
has its own rewrites, e.g. `monoidal.Rewrite = Rewrite[monoidal.Diagram]`, its
`Diagram.rewrite_factory`, and the steps of its axioms live in its module: `monoidal.Interchange`,
`braided.Slide`. What tells two
steps with the same boundary apart, e.g. the side an interchanger winds, is an argument of the
step. A user's own 2-cells are subclasses of `Step`.

Rewrites form a sesquicategory: vertical composition, whiskering and tensor, with strict
equality and no interchange law. The axioms of the free monoidal 2-category are not checked.

`Diagram.normalization` returns the `Rewrite` to the normal form, and `Rewrite.to_gif` animates a
rewrite: a pseudofunctor into animations. Each step draws itself (`Step.frame`), and between two
drawings of the same diagram that differ, e.g. the frame of a layer and the drawing of its
domain, the animation plays the laxator, moving every node vertically then horizontally.
`Equation.path` and `Equation.to_gif` are gone. `Drawing.interpolate` stays, for animations the
theory does not give.

## Consequences

- Every move of a gif of a rewrite is data: the Eckmann–Hilton loop is two `Interchange`s with
  `left=True`, not a guess.
- Two rewrites that differ by the interchange law, e.g. two steps one after the other or at
  once, are different rewrites; deciding equality modulo the Gray axioms is left to a later
  change.
- A laxator split in two halves keeps each node on the same side of a straight wire, which a
  single blend of both coordinates does not; that is not a proof for curved wires.
- The property matrix does not check rewrites: a step goes between parallel diagrams only, and
  the matrix draws each object of a law with its own boundary, not from one hom.
- `rigid.Unsnake` is the step of `snake_equations`; the braided Reidemeister moves are not yet
  `Step` subclasses, and a normalisation step that no `Diagram.rewrite_to` recognises is a plain
  `Step` named after `normalize`, drawn by `Drawing.interpolate`.

## Status

Accepted 2026-11-25, carried out by [#706](https://github.com/rel-int/discopy-next/pull/706).
