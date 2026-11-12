"""
The differential test of the Lean model of :mod:`discopy.egraph` against the
Python: Hypothesis draws the cases of each seed, the ``oracle`` binary that
``lake build`` makes answers them in one run, and each answer is compared.
The oracle knows nothing of diagrams: a box is the number of its first
occurrence in a history, whether it is deterministic and how many outputs it
has, and an e-graph its boxes on their wires with the class of each spider.

It is skipped when the binary is not built. ``ORACLE_EXAMPLES`` is the number
of cases per seed, 2,000 by default, and ``ORACLE_SEEDS`` the seeds separated
by commas, ``0`` by default.
"""

import json
import os
import pathlib
import subprocess
from collections.abc import Callable

import pytest
from hypothesis import HealthCheck, Phase, given, seed, settings
from hypothesis import strategies as st

from discopy.egraph import EGraph, History, Markov, partition

ORACLE = pathlib.Path(__file__).parent / ".lake" / "build" / "bin" / "oracle"
EXAMPLES = int(os.environ.get("ORACLE_EXAMPLES", "2000"))
SEEDS = tuple(map(int, os.environ.get("ORACLE_SEEDS", "0").split(",")))


def drawn(strategy: st.SearchStrategy, random_seed: int) -> list:
    """ The cases Hypothesis generates from a strategy under a seed. """
    cases = []

    @seed(random_seed)
    @settings(max_examples=EXAMPLES, database=None, deadline=None,
              phases=(Phase.generate, ),
              suppress_health_check=(HealthCheck.too_slow, ))
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


def encoding(history: History) -> tuple[dict, Callable[[EGraph], dict]]:
    """ A history as a case of the oracle, and what the oracle gives for an
    e-graph, with the boxes numbered by their first occurrence. """
    heads: dict = {}

    def nodes(graph: EGraph) -> list:
        return [
            [heads.setdefault(box, len(heads)), box.is_deterministic,
             list(inputs), list(outputs)]
            for box, (inputs, outputs) in zip(
                graph.graph.boxes, graph.graph.box_wires)]

    def operation(name, *args) -> list:
        if name != "add":
            return [name, *args]
        box, inputs = args
        return [name, heads.setdefault(box, len(heads)),
                box.is_deterministic, len(box.cod), list(inputs)]

    def result(graph: EGraph) -> dict:
        return {"classes": list(graph.classes), "nodes": nodes(graph)}

    start = history.graph
    return {
        "spiders": start.graph.n_spiders, **result(start),
        "operations": [operation(*step) for step in history.operations]
    }, result


@pytest.mark.parametrize("random_seed", SEEDS)
def test_partition(random_seed):
    cases = drawn(partitions(), random_seed)
    for case, result in zip(cases, answers(cases), strict=True):
        assert tuple(result["partition"]) == partition(
            case["spiders"], case["pairs"]), case
    print(f"partition: {len(cases)} cases agree under seed {random_seed}, "
          f"{len(set(map(json.dumps, cases)))} distinct")


@pytest.mark.parametrize("random_seed", SEEDS)
def test_fold(random_seed):
    histories = drawn(History[Markov].strategy(), random_seed)
    cases, results = zip(*map(encoding, histories))
    for history, case, result, answer in zip(
            histories, cases, results, answers(cases), strict=True):
        assert answer == result(history.fold()), case
    print(f"fold: {len(cases)} cases agree under seed {random_seed}, "
          f"{len(set(map(json.dumps, cases)))} distinct, "
          f"{sum(len(case['operations']) for case in cases)} operations")
