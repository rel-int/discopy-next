`neural.act` keeps a float64 network's losses and halt logits in float64;
low-precision readouts are still lifted to float32 ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
