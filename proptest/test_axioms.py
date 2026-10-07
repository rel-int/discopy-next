""" The property matrix: every law of every enrolled type, see
:meth:`discopy.axioms.Testable.matrix`. """

import pytest
from hypothesis import Phase, given, note, settings
from hypothesis import strategies as st

from conftest import PROFILE
from discopy.axioms import Testable
from discopy.utils import factory_name


def cells():
    """ One parameter per law, skipped when it does not apply and a
    strict expected failure when it is declared broken. """
    for law in Testable.matrix(once=PROFILE == "fast"):
        reason = (law.__doc__ or "").strip()
        if not law.parameters and law() is NotImplemented:
            marks = pytest.mark.skip(reason=reason)
        elif law.broken:
            marks = pytest.mark.xfail(reason=reason, strict=True)
        else:
            marks = ()
        yield pytest.param(
            law, marks=marks, id=f"{factory_name(law.bound)}.{law.name}")


@pytest.mark.parametrize("law", cells())
def test_axiom(law):
    """ Check a law on generated equations, a broken one up to its first
    counterexample, which :meth:`discopy.axioms.Axiom.falsify` shrinks on
    demand. """
    phases = (Phase.explicit, Phase.reuse, Phase.generate)\
        if law.broken else tuple(Phase)

    @settings(phases=phases)
    @given(data=st.data())
    def check(data):
        equation = data.draw(law.strategy(), label=law.name)
        note(equation)
        assert equation

    check()
