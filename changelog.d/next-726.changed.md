`split_memory` and `memory_size` move from `abc.MonoidalCategory` to
`abc.FeedbackCategory`, so a monoidal category that is not a feedback
category, e.g. `Matrix`, no longer has them, and the function
`abc.keyword_form` is removed for the static method
`abc.FeedbackCategory.keyword_form`; a class that traces without being a
feedback category, e.g. `Matrix`, still takes the keyword form of `trace`. The architecture diagram of
`discopy.abc` shows the syntax alone, and the docs list `discopy.egraph`
under semantics ([next#726](https://github.com/rel-int/discopy-next/pull/726)).
