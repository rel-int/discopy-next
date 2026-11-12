"""
The differential test of the Lean model of diagrams against
:class:`discopy.monoidal.Diagram`: Hypothesis draws diagrams with one box per
layer and no colours, as :meth:`Diagram.encode` writes them, the ``oracle``
binary composes, tensors and whiskers them in one run, and each answer is
compared with the Python's.

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

from discopy.monoidal import Box, Diagram, Ty

ORACLE = pathlib.Path(__file__).parent / ".lake" / "build" / "bin" / "oracle"
EXAMPLES = int(os.environ.get("ORACLE_EXAMPLES", "2000"))
SEEDS = tuple(map(int, os.environ.get("ORACLE_SEEDS", "0").split(",")))

types = st.lists(st.sampled_from("xyz"), max_size=3)


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
def diagrams(draw, dom: list[str]) -> dict:
    """ Up to four boxes, each on up to two wires of the codomain so far. """
    cod, boxes = dom, []
    for _ in range(draw(st.integers(0, 4))):
        offset = draw(st.integers(0, len(cod)))
        width = draw(st.integers(0, min(2, len(cod) - offset)))
        box = {"name": draw(st.sampled_from("fg")), "offset": offset,
               "dom": cod[offset:offset + width],
               "cod": draw(st.lists(st.sampled_from("xyz"), max_size=2))}
        cod = cod[:offset] + box["cod"] + cod[offset + width:]
        boxes.append(box)
    return {"dom": dom, "cod": cod, "boxes": boxes}


@st.composite
def cases(draw) -> dict:
    """ Two diagrams to compose or tensor, or one and a type to whisker. """
    op = draw(st.sampled_from(
        ["then", "tensor", "whisker_left", "whisker_right"]))
    f = draw(diagrams(draw(types)))
    if op == "then":
        return {"op": op, "f": f, "g": draw(diagrams(f["cod"]))}
    if op == "tensor":
        return {"op": op, "f": f, "g": draw(diagrams(draw(types)))}
    return {"op": op, "f": f, "ob": draw(types)}


def decode(diagram: dict) -> Diagram:
    """ The diagram of the Python that a case or an answer encodes. """
    return Diagram.decode(Ty(*diagram["dom"]), [
        (Box(box["name"], Ty(*box["dom"]), Ty(*box["cod"])), box["offset"])
        for box in diagram["boxes"]], cod=Ty(*diagram["cod"]))


def python(case: dict) -> Diagram:
    """ What the Python gives for a case. """
    f = decode(case["f"])
    return {
        "then": lambda: f >> decode(case["g"]),
        "tensor": lambda: f @ decode(case["g"]),
        "whisker_left": lambda: Ty(*case["ob"]) @ f,
        "whisker_right": lambda: f @ Ty(*case["ob"])}[case["op"]]()


@pytest.mark.parametrize("random_seed", SEEDS)
def test_premonoidal(random_seed):
    drawn_cases = drawn(cases(), random_seed)
    results = answers(drawn_cases)
    assert len(results) == len(drawn_cases)
    for case, result in zip(drawn_cases, results):
        expected = python(case)
        assert decode(result) == expected, case
        assert [box["offset"] for box in result["boxes"]] == list(
            expected.offsets), case
    print(f"premonoidal: {len(drawn_cases)} cases agree under seed "
          f"{random_seed}, {len(set(map(json.dumps, drawn_cases)))} distinct")
