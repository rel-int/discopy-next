`kleisli.additive.unpack_value` and `kleisli.multiplicative.unpack_value`
take the packed type, so a channel on a single wire passes a `Tagged` or
`Row` value through whole instead of stripping, re-tagging or splatting it
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
