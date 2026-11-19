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


def test_Function_no_type_checking_of_a_subclass_is_scoped():
    from discopy.python import multiplicative
    bad = multiplicative.Function(str, int, int)
    with multiplicative.Function.no_type_checking:
        assert bad(1) == "1"
        with raises(TypeError):
            Function(str, int, int)(1)
    with raises(TypeError):
        bad(1)
    with Function.no_type_checking:
        with multiplicative.Function.no_type_checking:
            assert bad(1) == "1"
    with raises(TypeError):
        bad(1)
    with Function.no_type_checking:
        assert bad(1) == "1"
    assert "type_checking" not in vars(multiplicative.Function)


def test_no_tree():
    from discopy.utils import dumps
    with raises(TypeError, match="A function has no tree"):
        dumps(Function(abs, int, int))
    assert "serialisation" not in Function.axioms


def test_then_flattens_its_own_compositions_only():
    from functools import reduce
    from discopy.python.function import Composition
    succ = Function(lambda x: x + 1, int, int)
    long = reduce(Function.then, 2000 * [succ])
    assert long(0) == 2000 and isinstance(long.inside, Composition)
    assert long.inside.factors == 2000 * (succ, )
    assert succ.then(succ.then(succ), succ).inside.factors == 4 * (succ, )
    assert succ.then() is succ

    class Polynomial:
        factors = (2, 3)

        def __call__(self, x):
            return 6 * x

    polynomial = Function(Polynomial(), int, int)
    assert (succ >> polynomial >> succ)(1) == 13
    assert (polynomial >> succ)(1) == 7 and (succ >> polynomial)(1) == 12
