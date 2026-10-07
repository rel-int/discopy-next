from pytest import raises

from discopy.traced import *
from discopy.utils import AxiomError


def test_trace_repr():
    trace = Box('f', 'x', 'x').trace()
    assert repr(trace) == "traced.Trace(traced.Box('f', monoidal.Ty(cat.Ob("\
        "'x')), monoidal.Ty(cat.Ob('x'))), left=False)"


def test_trace_error():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_trace_dagger():
    f = Box('f', 'x', 'x')
    assert f.trace().dagger() == f.dagger().trace()


def test_trace_vanishing():
    from discopy import compact, matrix, ribbon
    from discopy.python import additive, multiplicative

    def vanishes(morphism):
        return morphism.trace(mem=morphism.dom[:0]) == morphism

    x = compact.Ty('x')
    f = compact.Box('f', x @ x, x @ x)
    assert all(map(vanishes, (
        f, f.to_hypergraph(), f.to_map(), f.to_drawing())))

    y = ribbon.Ty('y')
    assert vanishes(ribbon.Box('g', y @ y, y @ y))
    assert vanishes(matrix.Matrix[bool].swap(1, 1))
    assert vanishes(additive.Function(
        lambda i, tag=0: (i, tag), (int, int), (int, int)))
    assert vanishes(multiplicative.Function(
        lambda i, j: (i, j), (int, int), (int, int)))
