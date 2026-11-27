`docs/notebooks/optyx/readme_example.md`, the old optyx README, executes
and is listed under "Optyx notebooks": its sweep of the contraction's `chi`
starts at 2, since at `chi=1` quimb's QR divides by zero ([next#442](https://github.com/rel-int/discopy-next/issues/442)), and
takes the median of three contractions per `chi` instead of ten. It leaves
`KNOWN_BROKEN`, which is now empty ([next#715](https://github.com/rel-int/discopy-next/pull/715)).
