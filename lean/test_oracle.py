"""
The differential test of the Lean model of :mod:`discopy.egraph` against the
Python: Hypothesis draws the cases of each seed, the ``oracle`` binary that
``lake build`` makes answers them in one run, and each answer is compared.

It is skipped when the binary is not built. ``ORACLE_EXAMPLES`` is the number
of cases per seed, 2,000 by default, and ``ORACLE_SEEDS`` the seeds separated
by commas, ``0`` by default.
"""

import json
import os
import pathlib
import subprocess

import pytest
from hypothesis import HealthCheck, Phase, given, seed, settings
from hypothesis import strategies as st

from discopy.egraph import partition

ORACLE = pathlib.Path(__file__).parent / ".lake" / "build" / "bin" / "oracle"
EXAMPLES = int(os.environ.get("ORACLE_EXAMPLES", "2000"))
SEEDS = tuple(map(int, os.environ.get("ORACLE_SEEDS", "0").split(",")))


def drawn(strategy: st.SearchStrategy, random_seed: int) -> list:
    """ The cases Hypothesis generates from a strategy under a seed. """
    cases = []

    @seed(random_seed)
    @settings(max_examples=EXAMPLES, database=None, deadline=None,
              phases=(Phase.generate, ),
              suppress_health_check=tuple(HealthCheck))
    @given(strategy)
    def collect(case):
        cases.append(case)

    collect()  # pylint: disable=no-value-for-parameter
    return cases


def answers(cases: list[dict]) -> list[dict]:
    """ What the Lean model gives for each case, in one run. """
    if not ORACLE.exists():
        pytest.skip(f"{ORACLE} is not built: run `lake build` in lean/")
    result = subprocess.run(
        [ORACLE], input="".join(json.dumps(case) + "\n" for case in cases),
        capture_output=True, text=True, check=True)
    return list(map(json.loads, result.stdout.splitlines()))


@st.composite
def partitions(draw) -> dict:
    """ Up to twelve spiders and pairs of them. """
    n = draw(st.integers(0, 12))
    spider = st.integers(0, n - 1) if n else st.nothing()
    return {"spiders": n, "pairs": draw(st.lists(
        st.tuples(spider, spider), max_size=2 * n))}


@pytest.mark.parametrize("random_seed", SEEDS)
def test_partition(random_seed):
    cases = drawn(partitions(), random_seed)
    results = answers(cases)
    assert len(results) == len(cases)
    for case, result in zip(cases, results):
        assert tuple(result["partition"]) == partition(
            case["spiders"], case["pairs"]), case
    print(f"partition: {len(cases)} cases agree under seed {random_seed}, "
          f"{len(set(map(json.dumps, cases)))} distinct")
