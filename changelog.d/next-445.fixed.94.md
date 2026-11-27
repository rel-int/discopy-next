`optyx.channel.Diagram.fix` warns that the result is truncated to `max_chi`
only when its backend compresses: `DiscopyBackend` and a `QuimbBackend`
without a compressed optimiser return the exact state without the false
warning ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
