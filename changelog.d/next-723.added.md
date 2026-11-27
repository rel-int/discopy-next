`discopy.lean` compiles `abc.FeedbackCategory`: a call with keywords is an
operation of its own, named after the method and its keywords, e.g.
`feedbackDomCodMem`, and a call leaving a numeric default out passes it,
so that `f.delay()` is `delay f 1` ([next#723](https://github.com/rel-int/discopy-next/pull/723)).
