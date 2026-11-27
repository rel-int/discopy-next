`neural.train.per_board_backward` drops a board whose gradient ratio is not
finite from the excluded (`halt.`) parameters too, so one NaN board no longer
turns every parameter NaN through `ActPool`'s global clip ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
