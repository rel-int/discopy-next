# -*- coding: utf-8 -*-

""" DisCoPy: the Python toolkit for computing with string diagrams. """

import doctest

from discopy import (
    testable,
    abc,
    cat,
    monoidal,
    spacial,
    braided,
    symmetric,
    biclosed,
    rigid,
    pivotal,
    ribbon,
    compact,
    markov,
    comarkov,
    closed,
    frobenius,
    hypergraph,
    ehypergraph,
    cmap,
    interaction,
    feedback,
    stream,
    para,
    lens,
    op,
    neural,
    python,
    kleisli,
    monad,
    matrix,
    tensor,
    hopf,
    group,
    quantum,
    grammar,
    drawing,
    utils,
    config,
    messages,
)

from discopy.version import (
    version as __version__,
    version_tuple as __version_info__
)

# A docstring example that needs an optional backend says so with `+EXTRA`.
# Registering the name keeps those docstrings parseable by plain doctest;
# acting on it is the job of `--skip-extra`, see test/plugin.py.
doctest.register_optionflag("EXTRA")
