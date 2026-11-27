Every push to `main` gets its own `build` and `benchmark` run, since a
concurrency group per ref let a newly queued push cancel the pending one
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
