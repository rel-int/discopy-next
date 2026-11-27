A push to `main` measures every Python of `build`'s matrix: with `fail-fast`
off, one version failing in its setup leaves the other two running, so
`Test notebooks` — which runs on a push to `main` and on no pull request —
still runs on the versions that survived. The composite setup action gives
each download 120 seconds, uv's default 30 having not been enough for the
55 MiB `egglog` wheel ([next#639](https://github.com/rel-int/discopy-next/pull/639)).
