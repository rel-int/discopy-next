`discopy.drawing.nicer`: nine hand features of a drawing, each a cost
(wire length, bends, slope, aspect, asymmetry, whitespace, misalignment,
label overlap, area), `hand_score` minus their weighted sum, with equal weights
until fitted on votes, and `make_nicer`, the foliation of a diagram whose
drawing scores best, or the diagram itself on a tie ([next#582](https://github.com/rel-int/discopy-next/pull/582)).
