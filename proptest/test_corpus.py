"""
The corpus: terms written by hand or by a language model, each aimed at a
shape the strategies of :mod:`discopy.testable` do not draw, replayed against
the cell of the property matrix it names, see ``proptest/corpus/``.

A module of the corpus defines ``CORPUS``, which maps a category to the
name of one of its laws, and that to entries keyed by what each one probes. An
entry is a thunk returning the arguments of the law by name, with the value
of each type variable its arguments do not show on their boundary, e.g. the
memory ``M`` of a trace. The other variables are read off the domain and
codomain of the morphisms, and every argument must then have the type its
premise asks for, as :func:`discopy.testable.inhabits` says. An entry that
fails this check is a bad entry, not a counterexample to the law.
"""

import importlib
import pkgutil
from typing import TypeVar, get_args, get_origin

import pytest
from hypothesis.errors import UnsatisfiedAssumption

from discopy.testable import AxiomFailure, Hom, Var, inhabits, substitute
from discopy.utils import factory_name

import proptest.corpus


def entries():
    """ One parameter per entry of every module of the corpus. """
    for module in pkgutil.iter_modules(proptest.corpus.__path__):
        corpus = importlib.import_module(
            f"proptest.corpus.{module.name}").CORPUS
        for category, laws in corpus.items():
            for name, probes in laws.items():
                for probe, thunk in probes.items():
                    yield pytest.param(
                        getattr(category, name), thunk,
                        id=f"{factory_name(category)}.{name}[{probe}]")


def scope(law, values):
    """
    The scope of the law at the values of an entry: its type variables those
    the entry gives, then those a premise is bare in, e.g. ``A`` and ``B`` in
    ``f: Hom[C1, A, B]`` and ``X`` in ``x: Var[C0, X]``.
    """
    annotations, given = law.annotations(), dict(values)
    for parameter in law.parameters:
        kind, value = annotations[parameter.name], values[parameter.name]
        bounds = {Hom: lambda: (value.dom, value.cod), Var: lambda: (value, )}
        bound = bounds.get(get_origin(kind), tuple)()
        for variable, ob in zip(get_args(kind)[1:], bound):
            if isinstance(variable, TypeVar):
                given.setdefault(variable.__name__, ob)

    def missing(name, _sort):
        raise KeyError(f"{law} needs the entry to give {name}.")
    return law.scope(missing, given)


@pytest.mark.parametrize("law, thunk", entries())
def test_corpus(law, thunk):
    """
    Check a law on the arguments of one entry, which must have the types its
    premises ask for, and a broken one up to the failure it declares.
    """
    values = thunk()
    arguments = {p.name: values[p.name] for p in law.parameters}
    environment, annotations = scope(law, values), law.annotations()
    for name, value in arguments.items():
        assert inhabits(value, substitute(annotations[name], environment)),\
            f"bad entry: {name} = {value!r} is not {annotations[name]}"
    try:
        verdict = law(**arguments)
    except AxiomFailure:
        if law.broken:
            pytest.xfail((law.__doc__ or "").strip())
        raise
    except UnsatisfiedAssumption:
        pytest.fail(f"bad entry: {law} assumes its arguments away")
    assert inhabits(verdict, law.output(environment))
    assert verdict
