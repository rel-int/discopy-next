"""
The property matrix: every law of every enrolled category, see
:meth:`discopy.axioms.Testable.matrix`.
"""

import pytest
from hypothesis import Phase, given, note, settings
from hypothesis import strategies as st

from discopy.axioms import Testable, inhabits

from proptest.conftest import PROFILE


def cells():
    """
    One parameter per law, skipped when it does not apply and a strict
    expected failure when it is declared broken.
    """
    for law in Testable.matrix(once=PROFILE == "fast"):
        reason = (law.__doc__ or "").strip()
        if not law.parameters and law() is NotImplemented:
            marks = pytest.mark.skip(reason=reason)
        elif law.broken:
            marks = pytest.mark.xfail(reason=reason, strict=True)
        else:
            marks = ()
        yield pytest.param(law, marks=marks, id=repr(law))


@pytest.mark.parametrize("law", cells())
def test_axiom(law):
    """
    Check a law against generated arguments, a broken one up to its first
    counterexample, which :meth:`discopy.axioms.Axiom.falsify` shrinks on
    demand, and the equation it states against the type of its output,
    evaluated in the scope its arguments were drawn in.
    """
    phases = (Phase.explicit, Phase.reuse, Phase.generate)\
        if law.broken else tuple(Phase)

    @settings(phases=phases)
    @given(data=st.data())
    def check(data):
        scope, arguments = data.draw(
            law.sequents(**law.params), label=law.name)
        verdict = law(**arguments)
        note(verdict)
        assert inhabits(verdict, law.output(scope))
        assert verdict

    check()
