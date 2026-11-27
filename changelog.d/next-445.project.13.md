`benchmark_comment.py` skips a pull request retargeted while its benchmark
ran, with a `::notice::`, and exits 0, instead of failing the job as a run
that does not belong to it, or, for a pull request from a fork, posting the
comparison against its old base; the next run posts ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
