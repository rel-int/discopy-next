> cubic-dev-ai review 5999582619 on dcaa0d6f:
>
> 1. discopy/drawing/picture.py:577 [P2] `to_dot` interpolates box names and wire labels into DOT/HTML without escaping them, so ordinary diagrams with quotes or HTML metacharacters cannot be laid out.
> 2. CHANGELOG.md:40 [P3] This entry doesn't say that `samples(graphviz=True)` needs the external Graphviz `dot` executable, and `layout` raises `RuntimeError` when it is missing.
> 3. docs/notebooks/from_picture.md:60 [P3] This runnable cell requires the external Graphviz `dot` executable, but the new section does not mention that prerequisite.
> 4. discopy/drawing/picture.py:506 [P3] Graphviz samples lay out every accepted diagram twice: `fits` calls `layout`, then `render` calls it again.

- [x] picture.py:577 escape names and labels in to_dot
- [x] CHANGELOG.md:40 say Graphviz is needed
- [x] from_picture.md:60 say Graphviz is needed
- [WIP] @T-a78eb6e6-2026-11-20 21:22 picture.py:506 lay each sample out once
