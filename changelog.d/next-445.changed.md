`feedback.Head` and `Tail`, at every level of the hierarchy, now have a
dagger: `Head(f).dagger() == Head(f.dagger())`, and the same for `Tail`,
where they used to raise `TypeError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
