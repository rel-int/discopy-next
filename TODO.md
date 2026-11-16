> Review round on #556 (chatgpt-codex-connector[bot]), head 5f986631:
>
> 1. discopy/csp.py:135 (thread 4588849802)
> **  Account for daggered boxes when extracting scopes**
>
> For diagrams containing a daggered, non-symmetric relation, concatenating inputs then outputs reverses the predicate's intended argument order relative to the tensor functor. For example, let `x` have size 2, interpret `lt` as `a < b`, and build `(frobenius.Spider(0, 1, x) >> frobenius.Box('lt', x, x).dagger()).to_hypergraph()`. Backtracking reports `(0, 1)`, projecting to output 1, while `language.functor(bool)` projects to output 0 because `cat.Functor.__call__` evaluates the original box and takes its dagger. Normalize daggered boxes and their scope ordering so that `holds` and `solutions` agree with the advertised functor semantics.
>
> Useful? React with 👍 / 👎.

- [x] discopy/csp.py:135 ** Account for daggered boxes when extracting scopes** For diagrams containing a daggered, non-symmetric rela…
