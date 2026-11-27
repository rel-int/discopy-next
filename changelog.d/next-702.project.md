The `proptest` workflow runs every cell at `pr`'s twenty examples under
`--hypothesis-seed=0` on `main` too, with one worker per vCPU, and no longer
runs nightly: `explore`'s thousand examples per cell had cancelled every run
of `main` at 30 minutes since the matrix grew past 2,900 cells, 4% of the
way through ([next#702](https://github.com/rel-int/discopy-next/pull/702)).
