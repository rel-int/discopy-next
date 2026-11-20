from pathlib import Path

from discopy import lean

ABC = Path(__file__).parents[1] / "discolean" / "Discolean" / "ABC.lean"


def test_abc_is_generated():
    "discolean/Discolean/ABC.lean is what `python -m discopy.lean` prints."
    assert ABC.read_text(encoding="utf-8") == lean.generate()
