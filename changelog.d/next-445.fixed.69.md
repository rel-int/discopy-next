`Matrix.is_close` gives its real `rtol` default (`1.e-8`), and
`tensor.Diagram.jacobian` says it returns a `Sum` diagram whose `eval` is
`Tensor.jacobian` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
