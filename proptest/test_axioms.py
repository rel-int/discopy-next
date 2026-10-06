""" Property tests for DisCoPy's principal categorical data structures. """

import pytest
from hypothesis import Phase, given, note, settings
from hypothesis import strategies as st

from conftest import PROFILE
from discopy.axioms import Axiom, Constant, Testable
from discopy.utils import factory_name


def types() -> tuple[type[Testable], ...]:
    """
    The testable types the matrix quantifies over: every transitive
    subclass of :class:`discopy.axioms.Testable` that generates its own
    terms, i.e. implements :meth:`discopy.axioms.Testable.strategy` —
    which is how a type enrols itself, so the matrix follows the
    package rather than a list kept beside it. A class stating no law
    gets no cell, nor does a category over a fixed vocabulary, whose
    :attr:`discopy.abc.Category.generators` are the
    :class:`discopy.search.Constant` rules of its words or gates: it
    fills only the sequents its vocabulary derives, not the ones a law
    draws, and its laws are those of the free category it lives in.
    """
    def generates(testable):
        try:
            testable.strategy()
        except NotImplementedError:
            return False
        generators = dict(getattr(testable, "generators", {}))
        return not any(
            isinstance(value, Constant) for value in generators.values())

    return tuple(sorted(
        (testable for testable in Testable.subclasses()
         if testable.axioms and generates(testable)), key=factory_name))


def declaring(testable: type[Testable], name: str) -> type:
    """
    The class whose declaration of the law of that name ``testable``
    inherits: the first in its MRO that has the name in its namespace, the
    one :meth:`discopy.axioms.Testable.declarations` binds.
    """
    return next(base for base in testable.__mro__ if name in base.__dict__)


def once_per_declaration(
        cells: list[tuple[type[Testable], Axiom]]
) -> list[tuple[type[Testable], Axiom]]:
    """
    One cell per declaration of a law, under the ``fast`` profile: the
    enrolled type nearest the class declaring it, so that a law is tested
    once bound to its defining class rather than again on every type
    inheriting it. A type restating an inherited law — broken, weakened or
    modulo a quotient — declares it anew, and is the nearest to that.
    """
    nearest = {}
    for testable, axiom in cells:
        owner = declaring(testable, axiom.name)
        distance = testable.__mro__.index(owner)
        if (owner, axiom.name) not in nearest\
                or distance < nearest[owner, axiom.name][0]:
            nearest[owner, axiom.name] = (distance, testable, axiom)
    return sorted(
        ((testable, axiom) for _, testable, axiom in nearest.values()),
        key=lambda cell: (factory_name(cell[0]), cell[1].name))


def axiom_parameters(broken: bool = False):
    """
    Translate every axiom of every testable type to a pytest parameter,
    one per declaration under the ``fast`` profile, the laws declared
    broken and the others apart. An axiom taking no argument states its
    verdict without one: :obj:`NotImplemented` skips the cell rather
    than generating arguments it could not satisfy.
    """
    cells = [
        (testable, axiom)
        for testable in types() for axiom in testable.axioms.values()]
    if PROFILE == "fast":
        cells = once_per_declaration(cells)
    for testable, axiom in cells:
        if axiom.broken != broken:
            continue
        if not axiom.parameters and axiom() is NotImplemented:
            marks = pytest.mark.skip(reason=axiom.__doc__.strip())
        elif axiom.broken:
            marks = pytest.mark.xfail(
                reason=axiom.__doc__.strip(), strict=True)
        else:
            marks = ()
        yield pytest.param(
            axiom, marks=marks, id=f"{factory_name(testable)}.{axiom.name}")


def check(axiom: Axiom, data: st.DataObject) -> None:
    """ Check an axiom of a testable type on a generated equation. """
    equation = data.draw(axiom.strategy(), label=axiom.name)
    note(equation)
    assert equation


@pytest.mark.parametrize("axiom", axiom_parameters())
@given(data=st.data())
def test_axiom(axiom, data):
    """ Check a law that is expected to hold. """
    check(axiom, data)


@pytest.mark.parametrize("axiom", axiom_parameters(broken=True))
@settings(phases=(Phase.explicit, Phase.reuse, Phase.generate))
@given(data=st.data())
def test_broken_axiom(axiom, data):
    """
    Check a law declared broken, expecting the failure: the first
    counterexample is enough, so the phases that shrink and explain it
    are left to :meth:`discopy.axioms.Axiom.falsify`, on demand.
    """
    check(axiom, data)
