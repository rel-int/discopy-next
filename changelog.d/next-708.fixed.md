`cat.TermBase.normal_form` contracts a `let` whose variable has the name of
a variable its expression uses, e.g. `let(f(a), lambda a: g(a))`, and the
property matrix draws such terms: a naming of the kappa terms may reuse a
name its expression uses up ([next#708](https://github.com/rel-int/discopy-next/pull/708)).
