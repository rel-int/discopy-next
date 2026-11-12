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

from benchmark.test_egraph import EMPTY, TRACES, load
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


def encoding(history: History) -> tuple[dict, Callable[..., dict]]:
    """ A history as a case of the oracle, and what the oracle gives for an
    e-graph and the outputs of each ``add``, with the boxes numbered by their
    first occurrence. """
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

    def result(graph: EGraph, outputs=()) -> dict:
        return {"classes": list(graph.classes), "nodes": nodes(graph),
                "outputs": list(map(list, outputs))}

    start = history.graph
    case = result(start) | {
        "spiders": start.graph.n_spiders,
        "operations": [operation(*step) for step in history.operations]}
    del case["outputs"]
    return case, result


def folded(history: History) -> tuple[EGraph, list]:
    """ The fold of a history with the outputs each ``add`` gave. """
    graph, outputs = history.graph, []
    for name, *args in history.operations:
        graph = getattr(graph, name)(*args)
        if name == "add":
            graph, added = graph
            outputs.append(added)
    return graph, outputs


def built(history: History) -> tuple[EGraph, list]:
    """ What the builder of a history freezes into with the outputs each
    ``add`` gave. """
    builder, outputs = history.graph.builder(), []
    for name, *args in history.operations:
        added = getattr(builder, name)(*args)
        if name == "add":
            outputs.append(added)
    return builder.freeze(), outputs


@pytest.mark.parametrize("random_seed", SEEDS)
def test_partition(random_seed):
    cases = drawn(partitions(), random_seed)
    for case, result in zip(cases, answers(cases), strict=True):
        assert tuple(result["partition"]) == partition(
            case["spiders"], case["pairs"]), case
    print(f"partition: {len(cases)} cases agree under seed {random_seed}, "
          f"{len(set(map(json.dumps, cases)))} distinct")


@pytest.mark.parametrize("random_seed", SEEDS)
def test_history(random_seed):
    histories = drawn(History[Markov].strategy(), random_seed)
    cases, results = zip(*map(encoding, histories))
    for history, case, result, answer in zip(
            histories, cases, results, answers(cases), strict=True):
        graph, outputs = folded(history)
        assert result(history.fold(), outputs) == result(graph, outputs), case
        assert answer["fold"] == result(graph, outputs), case
        graph, outputs = built(history)
        assert result(history.build(), outputs) == result(graph, outputs), case
        assert answer["build"] == result(graph, outputs), case
    print(f"history: {len(cases)} cases agree under seed {random_seed}, "
          f"{len(set(map(json.dumps, cases)))} distinct, "
          f"{sum(len(case['operations']) for case in cases)} operations")


def replayed(trace, twice: bool) -> History:
    """ The history of a trace of ``benchmark/data/egraph-traces`` through a
    builder, each e-node added then related to the class recorded for it,
    and ``twice`` on base classes of its own, related then rebuilt. """
    builder, operations = EMPTY.builder(), []

    def replay(boxes) -> list:
        classes = []
        for head, out, *args in trace.rows:
            inputs = tuple(classes[i] for i in args)
            operations.append(("add", boxes[head], inputs))
            spider, = builder.add(boxes[head], inputs)
            if out == len(classes):
                classes.append(spider)
                continue
            operations.append(("merge", spider, classes[out]))
            builder.union(spider, classes[out])
        return classes

    classes = replay(trace.boxes)
    if twice:
        others = replay(trace.others)
        operations.extend(
            ("merge", *pair) for pair in zip(classes[:trace.bases], others))
        operations.append(("rebuild", ))
    return History[Markov](EMPTY, operations)


@pytest.mark.parametrize("twice", (False, True))
def test_traces(twice):
    index = json.loads((TRACES / "index.json").read_text())["traces"]
    histories = [
        replayed(load(chosen, n + 1), twice)
        for chosen in ("random", "largest")
        for n in range(sum(item["chosen"] == chosen for item in index))]
    cases, results = zip(*map(encoding, histories))
    cases = [case | {"run": ["build"]} for case in cases]
    for history, result, answer in zip(
            histories, results, answers(cases), strict=True):
        graph, outputs = built(history)
        assert result(history.build(), outputs) == result(graph, outputs)
        assert answer["build"] == result(graph, outputs)
    print(f"traces: {len(cases)} agree, twice is {twice}, "
          f"{sum(len(case['operations']) for case in cases)} operations")
