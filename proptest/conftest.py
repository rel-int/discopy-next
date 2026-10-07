"""
Hypothesis profiles for the property matrix, selected by
``HYPOTHESIS_PROFILE``: ``dev`` by default, ``fast`` to develop with —
derandomized, one cell per declaration of a law — ``pr`` for the small
budget of a pull request and ``explore`` for the large one of ``main``
and the nightly run. All but ``fast`` share the example database that CI
downloads before a run and uploads after it.
"""

import os

import matplotlib
from hypothesis import settings
from hypothesis.database import DirectoryBasedExampleDatabase

matplotlib.use("Agg")

PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "dev")

COMMON = dict(
    derandomize=False, deadline=None, print_blob=True,
    database=DirectoryBasedExampleDatabase(".hypothesis/examples"))

settings.register_profile("pr", max_examples=20, **COMMON)
settings.register_profile("explore", max_examples=1000, **COMMON)
settings.register_profile("dev", max_examples=100, **COMMON)
settings.register_profile(
    "fast", max_examples=100, derandomize=True, deadline=None,
    print_blob=True)
settings.load_profile(PROFILE)
