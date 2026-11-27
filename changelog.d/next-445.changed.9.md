`cat.Transformation.id` has the shared `utils.identity` as its components,
and `then` drops it on either side and composes components given by a
dictionary into a dictionary, so that the identity is a unit and composition
is associative up to `==`, as for `cat.Functor` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
