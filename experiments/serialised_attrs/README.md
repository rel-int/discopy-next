# serialised_attrs on every class: a measurement, not a proposal

This branch converts every class that hand-writes `to_tree`, `from_tree` or `__repr__` to the
`Serialisable.serialised_attrs` declaration of discopy/discopy#744 where the declaration can express
it, and leaves the hand-written methods where it cannot, so that what breaks can be measured.

To reproduce, from a checkout of next `main` (TREE_MAIN) and of this branch (TREE_EXP):

    CAPTURE_OUT=main.json PYTHONPATH=experiments/serialised_attrs \
        uv run pytest -q -p dcp_sampler                     # in TREE_MAIN: sample the suite's objects
    python -I compare.py TREE_MAIN main.json base.json      # the noise floor: main against itself
    python -I compare.py TREE_EXP main.json exp.json        # this branch against main's objects
    python -I forward_err.py TREE_MAIN exp.json main.json   # main reading this branch's trees
