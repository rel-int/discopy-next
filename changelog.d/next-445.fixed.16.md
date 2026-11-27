`Diagram.normal_form` and `normalize` read the diagram's boxes and offsets
once per sweep, and `Diagram.boxes` is linear, so normalising an
already-normal 800-box diagram takes about 1ms instead of 1.4s ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
