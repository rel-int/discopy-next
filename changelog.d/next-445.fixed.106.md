`to_tk` translates `Measure(override_bits=True)` after classical
post-processing correctly: it discards the old bits in post-processing and
measures into fresh ones, where before the post-processing read the new value
and a `Copy` before it raised `IndexError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
