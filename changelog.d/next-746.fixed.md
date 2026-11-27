The API docs show short names: a page is titled with the last part of its
name, which it lost when its package did not import it, e.g.
`neural.cell`, module tables and references to modules and objects drop
the `discopy.` prefix, and the bases of a class name only as much of their
path as tells two of them apart, e.g. `pivotal.Diagram, balanced.Diagram`
for `ribbon.Diagram`, as references in one paragraph do; `test/api_names.py`
checks both. `discopy.drawing` has
its own page instead of appearing twice under semantics, and the subpackages
follow it in the order lambek, neural, optyx, quantum ([next#746](https://github.com/rel-int/discopy-next/pull/746)).
