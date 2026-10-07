"""
Measure how the search of :meth:`discopy.monoidal.Diagram.strategy` samples:
the Hypothesis statistics (valid and invalid examples, time per example), the
dead ends the search retries, and the distribution of what it samples.

Each cell draws ``--examples`` diagrams of one category, either toward no
goal or toward a fixed goal ``dom -> cod``, under a derandomised Hypothesis
with no database, so that two checkouts measure the same budget::

    uv run python proptest/search_experiment.py --examples 500 --out run.json

The JSON maps each cell to its statistics; ``--compare base.json`` prints
the two runs side by side.
"""

import argparse
import json
import statistics
import time
from collections import Counter

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis.internal.observability import with_observability_callback

from discopy import (
    biclosed, braided, closed, compact, feedback, frobenius, markov,
    monoidal, pivotal, rigid, symmetric)

PLUMBING = {
    "Swap", "Permutation", "Braid", "Copy", "Merge", "Discard", "Spider",
    "Cup", "Cap", "Twist"}

LEVELS = {
    "monoidal": monoidal, "braided": braided, "symmetric": symmetric,
    "markov": markov, "biclosed": biclosed, "closed": closed,
    "rigid": rigid, "pivotal": pivotal, "compact": compact,
    "frobenius": frobenius, "feedback": feedback}

GOALS = {
    "swap": ("ab", "ba"),
    "copy": ("a", "aa"),
    "discard": ("ab", "a"),
    "cycle": ("abc", "cab"),
    "merge": ("aa", "a"),
}


def goal_of(module, name):
    """ The types of a named goal at a level. """
    dom, cod = GOALS[name]
    return module.Ty(*dom), module.Ty(*cod)


def dead_end_class():
    """ The exception the search raises on a dead end, wherever it lives. """
    from discopy import abc, pattern
    return getattr(pattern, "DeadEnd", None) or abc.DeadEnd


def measure(module, goal, examples):
    """ The statistics of one cell. """
    category = module.Diagram
    dom, cod = goal_of(module, goal) if goal else (None, None)
    strategy = category.strategy(dom=dom, cod=cod)
    samples, statuses, dead_ends = [], Counter(), Counter()
    reasons = Counter()
    DeadEnd = dead_end_class()
    original = DeadEnd.__init__

    def counting(self, *args):
        dead_ends["raised"] += 1
        original(self, *args)

    def observe(observation):
        if getattr(observation, "type", None) == "test_case":
            statuses[observation.status] += 1
            if observation.status == "gave_up":
                reasons[observation.status_reason[:80]] += 1

    @settings(max_examples=examples, database=None, derandomize=True,
              deadline=None, phases=[Phase.generate],
              suppress_health_check=list(HealthCheck))
    @given(strategy)
    def sample(diagram):
        samples.append(diagram)

    DeadEnd.__init__ = counting
    start = time.perf_counter()
    try:
        with with_observability_callback(observe):
            sample()
        error = None
    except Exception as exception:  # pylint: disable=broad-exception-caught
        error = f"{type(exception).__name__}: {exception}"[:200]
    finally:
        DeadEnd.__init__ = original
    elapsed = time.perf_counter() - start
    result = summarise(samples, statuses, dead_ends, elapsed, error)
    result["invalid_reasons"] = dict(reasons.most_common(5))
    return result


def summarise(samples, statuses, dead_ends, elapsed, error):
    """ The statistics of the samples of one cell. """
    boxes = [len(diagram.boxes) for diagram in samples]
    plumbing = [sum(type(box).__name__ in PLUMBING for box in diagram.boxes)
                for diagram in samples]
    kinds = Counter(
        type(box).__name__ for diagram in samples for box in diagram.boxes)
    shapes = {repr(diagram) for diagram in samples}
    total = sum(statuses.values()) or 1

    def mean(values):
        return round(statistics.mean(values), 3) if values else None

    return {
        "valid": statuses["passed"], "invalid": statuses["gave_up"],
        "invalid_ratio": round(statuses["gave_up"] / total, 3),
        "dead_ends": dead_ends["raised"],
        "dead_ends_per_valid": round(
            dead_ends["raised"] / max(statuses["passed"], 1), 3),
        "ms_per_example": round(1000 * elapsed / total, 2),
        "boxes": mean(boxes), "max_boxes": max(boxes, default=0),
        "layers": mean([len(diagram) for diagram in samples]),
        "plumbing_ratio": round(
            sum(plumbing) / max(sum(boxes), 1), 3),
        "identities": round(
            sum(not b for b in boxes) / max(len(boxes), 1), 3),
        "distinct": len(shapes),
        "kinds": dict(kinds.most_common()),
        "error": error}


CELLS = [(level, None) for level in LEVELS] + [
    (level, goal) for level in (
        "monoidal", "symmetric", "markov", "compact", "frobenius")
    for goal in GOALS]

COLUMNS = ("valid", "invalid_ratio", "dead_ends_per_valid", "ms_per_example",
           "boxes", "plumbing_ratio", "identities", "distinct")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=300)
    parser.add_argument("--out")
    parser.add_argument("--compare")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    results = {}
    for level, goal in CELLS:
        name = f"{level}:{goal or '-'}"
        if args.only and args.only not in name:
            continue
        results[name] = measure(LEVELS[level], goal, args.examples)
        print(name, {key: results[name][key] for key in COLUMNS},
              results[name]["error"] or "", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=1)
    if args.compare:
        with open(args.compare, encoding="utf-8") as file:
            base = json.load(file)
        print(f"{'cell':22}" + "".join(f"{c[:14]:>16}" for c in COLUMNS))
        for name, row in results.items():
            before = base.get(name, {})
            print(f"{name:22}" + "".join(
                f"{str(before.get(c)) + '>' + str(row[c]):>16}"
                for c in COLUMNS))


if __name__ == "__main__":
    main()
