`optyx.compiler`'s summary renders `compile_to_semm`, `decompile_from_semm`
and `pattern_satisfies_order` as the functions they are, with their own
autosummary block, where `class.rst` used to document all three as classes;
so do the summaries of `compiler.semm`, which listed `compile_to_semm` and
`decompile_from_semm` the same way, and `compiler.semm_decompiler`, which
listed `decompile_to_fusion_network` ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
