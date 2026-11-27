An optyx box is what it computes: `channel.Channel.semantic_attrs` names
the attributes besides `name`, `dom`, `cod` and `data` that decide it, and
`setoid` compares them — a gate its `array`, which is what `is_conj` and
`is_gate_dagger` decide, a creation the internal states of its photons, a
`CQMap` its density matrix and a `BinaryMatrixBox` its matrix. Two boxes
computing different arrays were equal, every `classical.BinaryMatrix`
among them ([next#508](https://github.com/rel-int/discopy-next/pull/508)).
