from pytest import raises

from discopy.python.function import *
from discopy.utils import AxiomError


def test_Function_category():
    succ = Function(lambda x: x + 1, int, int)
    double = Function(lambda x: 2 * x, int, int)
    assert succ.dom == succ.cod == Ty(int)
    assert succ.then() is succ
    assert (succ >> double)(1) == 4 and (double >> succ)(1) == 3
    for x in (0, 7):
        assert (Function.id(int) >> succ)(x) == succ(x)\
            == (succ >> Function.id(int))(x)
        assert ((succ >> double) >> succ)(x) == (succ >> (double >> succ))(x)
    with raises(AxiomError):
        succ >> Function(len, str, int)


def test_Function_type_checking():
    succ = Function(lambda x: x + 1, int, int)
    with raises(TypeError):
        succ("1")
    with raises(TypeError):
        Function(str, int, int)(1)
    with raises(ZeroDivisionError):
        with Function.no_type_checking:
            assert Function(str, int, int)(1) == "1"
            1 / 0
    assert Function.type_checking
