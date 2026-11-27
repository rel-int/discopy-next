Kleisli channels of different monads raise `TypeError` when composed with
`>>` or tensored with `@`, instead of composing silently or failing only at
call time ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
