"""
The property matrix: every law of every enrolled category, see
:meth:`discopy.axioms.Testable.matrix`.
"""

import pytest
from hypothesis import Phase, given, note, settings
from hypothesis import strategies as st

from discopy.axioms import Testable, inhabits, position
from discopy.utils import factory_name

from proptest.conftest import PROFILE


def cells():
    """
    One parameter per law, skipped when it does not apply and a strict
    expected failure when it is declared broken.
    """
    for law in Testable.matrix(once=PROFILE == "fast"):
        reason = (law.__doc__ or "").strip()
        if not law.applicable:
            marks = pytest.mark.skip(reason=reason)
        elif law.broken:
            marks = pytest.mark.xfail(reason=reason, strict=True)
        else:
            marks = ()
        yield pytest.param(law, marks=marks, id=repr(law))


@pytest.mark.cell
@pytest.mark.parametrize("law", cells())
def test_axiom(law, drawn):
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
        drawn.add(repr(arguments))
        verdict = law(**arguments)
        note(verdict)
        assert inhabits(verdict, law.output(scope))
        assert verdict

    check()


@pytest.mark.parametrize(
    "category", Testable.enrolled(), ids=factory_name)
def test_position(category):
    """
    Every ``C0`` and ``C1`` above an enrolled category stands for its
    objects and its arrows, as :meth:`discopy.abc.Category.parameters` says,
    whichever class states it and however its bases pass it on, so that a
    sequent bound to the category reads as one of its own.
    """
    for klass in category.__mro__:
        for parameter in getattr(klass, "__type_params__", ()):
            if parameter.__name__ in ("C0", "C1"):
                assert position(parameter, category)\
                    == int(parameter.__name__[1]), (klass, parameter)
