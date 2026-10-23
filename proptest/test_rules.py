"""
The rules of every enrolled category, see
:meth:`discopy.axioms.Testable.matrix`.
"""

import pytest
from hypothesis import Phase, given, note, settings
from hypothesis import strategies as st

from discopy.axioms import Rule, Testable, inhabits

from proptest.conftest import PROFILE


def cells():
    """
    One parameter per rule, skipped when it does not apply and a strict
    expected failure when it is declared broken.
    """
    for rule in Testable.matrix(once=PROFILE == "fast", kind=Rule):
        reason = (rule.__doc__ or "").strip()
        if not rule.applicable:
            marks = pytest.mark.skip(reason=reason)
        elif rule.broken:
            marks = pytest.mark.xfail(reason=reason, strict=True)
        else:
            marks = ()
        yield pytest.param(rule, marks=marks, id=repr(rule))


@pytest.mark.cell
@pytest.mark.parametrize("rule", cells())
def test_rule(rule, drawn):
    """
    Apply a rule to the premises it draws for a goal, a broken one up to
    its first counterexample, and check its conclusion against the type its
    signature states, evaluated in the scope of the draw. The objects of a
    monoid are its one object, ``None``.
    """
    phases = (Phase.explicit, Phase.reuse, Phase.generate)\
        if rule.broken else tuple(Phase)
    ob = rule.category.ob
    objects = st.none() if ob is type(None) else ob.strategy()

    @settings(phases=phases)
    @given(data=st.data())
    def check(data):
        dom = data.draw(objects, label="dom")
        cod = data.draw(objects, label="cod")
        scope, premises = data.draw(
            rule.sequents(rule.given(dom, cod), **rule.params),
            label=rule.name)
        drawn.add(repr(premises))
        conclusion = rule(**premises)
        note(conclusion)
        assert inhabits(conclusion, rule.output(scope))

    check()
