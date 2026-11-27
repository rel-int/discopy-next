`build`'s Coverage step combines with `--debug=dataio`, naming each data file
it merges, and fails with its own error when it merged fewer than two: the
controller of `pytest -n auto` writes one and each worker one more, so the
controller's alone measured no test and read as a 36% fail-under regression.
`.github/tests/build_coverage.py` runs the step with and without the
workers' files ([next#596](https://github.com/rel-int/discopy-next/issues/596)).
