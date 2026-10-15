"""
Hypothesis profiles for the property matrix, selected by
``HYPOTHESIS_PROFILE``: ``dev`` by default, ``fast`` to develop with —
derandomized, one cell per declaration of a law — ``pr`` for the small
budget of a pull request and ``explore`` for the large one of ``main``
and the nightly run. All but ``fast`` share the example database that CI
downloads before a run and uploads after it.

Under ``CI`` a registered profile inherits Hypothesis's ``ci`` defaults,
``derandomize=True`` and hence ``database=None``, so both are explicit.
"""

import os

from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase

PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "dev")

COMMON = dict(
    deadline=None, print_blob=True,
    suppress_health_check=[HealthCheck.filter_too_much])

SHARED = dict(
    COMMON, derandomize=False,
    database=DirectoryBasedExampleDatabase(".hypothesis/examples"))

settings.register_profile("pr", max_examples=20, **SHARED)
settings.register_profile("explore", max_examples=1000, **SHARED)
settings.register_profile("dev", max_examples=100, **SHARED)
settings.register_profile(
    "fast", max_examples=100, derandomize=True, database=None, **COMMON)
settings.load_profile(PROFILE)
