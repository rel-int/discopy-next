`Matrix.id` builds its array over `int` and casts it with the active
backend, so a matrix inferred under `backend("pytorch")` traces, and
`Matrix.then` casts both factors to the dtype they promote to, so it
composes with the integer structural matrices on a backend whose product
does not promote ([next#583](https://github.com/rel-int/discopy-next/pull/583)).
