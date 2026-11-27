The `docs` extra installs `discopy[optyx]` and the `notebooks` extra joblib,
so with the `docs` group 13 of the 15 optyx notebooks render instead of
falling back to link pages, the other two being `KNOWN_BROKEN`; the docs
job's timeout rises to 45 minutes for them. The `dev` group installs
egglog, so the 40 egglog checks of `test/ehypergraph.py` and its fixture
run instead of skipping ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
