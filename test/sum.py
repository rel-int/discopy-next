import pytest

from discopy import cat, monoidal, symmetric
from discopy.sum import Sum


@pytest.mark.parametrize("module, new", [
    (cat, Sum), (monoidal, Sum[monoidal.Diagram])])
def test_deprecated_Sum_is_the_new_class(module, new):
    with pytest.deprecated_call(match=f"{module.__name__}.Sum is deprecated"):
        assert module.Sum is new


def test_Sum_is_a_box_of_its_category():
    f = monoidal.Box('f', 'x', 'x')
    assert monoidal.Diagram.sum_factory is Sum[monoidal.Diagram]
    assert issubclass(Sum[monoidal.Diagram], monoidal.Box)
    assert type(f + f) is Sum[monoidal.Diagram]
    assert (f + f).category is monoidal.Diagram
    assert type(cat.Box('g', 'x', 'y').zero('x', 'y')) is Sum


def test_star_import_keeps_the_builtin_sum():
    namespace = {}
    exec("from discopy import *", namespace)
    assert "sum" not in namespace and "cat" in namespace


@pytest.mark.parametrize("Diagram", [monoidal.Diagram, symmetric.Diagram])
def test_Sum_is_drawn_as_a_diagram(Diagram):
    f = Diagram.box_factory('f', 'x', 'x')
    assert (f + f).factory is Diagram
    assert (f + f).to_drawing() == Diagram.to_drawing(f + f)
    assert [box.name for box in (f + f).to_drawing().boxes] == ['f', '+', 'f']
