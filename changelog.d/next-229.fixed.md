`optyx.core.path.Matrix.prob_with_perceval` computes the Born rule of the
amplitudes, as its docstring and `prob`'s other branch say, where it used to
answer a probability conditional on the selections: two photons bunching
into one arm of a balanced beam splitter read `1` instead of `0.5`.
Perceval's own post-selection renormalises over the outcomes that survive
it, so the selections now pick out those outcomes instead, and the
`normalisation` that `dilate` leaves outside the unitary is put back on the
probabilities. It read right exactly where the selections carry all of the
mass or none of it, which both of its doctests did. As a tensor it also
raised `IndexError` for every diagram with a selection, giving
`amplitudes_2_tensor` one basis state per mode of the circuit where the
matrix has one column per *selected* outcome; the codomain is now the
occupations of the modes the selections leave open, as `eval` builds it, and
a selection of more photons than there are raises the same
`ValueError` as `eval` where it used to enumerate a negative photon count
([next#229](https://github.com/rel-int/discopy-next/issues/229)).
