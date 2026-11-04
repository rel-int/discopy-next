# -*- coding: utf-8 -*-

""" DisCoPy utility functions. """

from __future__ import annotations

import builtins
import inspect
import json
import sys
from fractions import Fraction
from functools import lru_cache, wraps
from math import ceil, prod
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Mapping,
    Iterable,
    Iterator,
    TypeVar,
    Any,
    Collection,
    NamedTuple,
    Sequence,
    Union,
    TYPE_CHECKING,
)

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from networkx import Graph, connected_components

import discopy.messages as messages

if TYPE_CHECKING:
    import numpy
    import sympy
    from discopy.monoidal import Ty
    from discopy.abc import Category

KT = TypeVar('KT')
VT = TypeVar('VT')
V2T = TypeVar('V2T')

_DEFAULT_FONT_PATH = (
    Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
_DEFAULT_FONT = FontProperties(fname=_DEFAULT_FONT_PATH)


class MappingOrCallable(Mapping[KT, VT]):
    """
    A Mapping or Callable object.

    Example
    -------
    T = MappingOrCallable[int, int]
    f = T(lambda x: x + 1)
    g = T({0: 1})
    assert f[0] == g[0] == 1
    assert len(g) == 1
    assert list(g) == [0]
    """
    def __class_getitem__(_, args: tuple[type, type]):
        source, target = args
        return (
            Mapping[source, target]  # ty: ignore[invalid-type-form]
            | Callable[[source], target])  # ty: ignore[invalid-type-form]

    def __init__(self, mapping) -> None:
        while isinstance(mapping, MappingOrCallable):
            mapping = mapping.mapping
        if not isinstance(mapping, Mapping) and not callable(mapping):
            raise TypeError(messages.NOT_MAPPING_OR_CALLABLE.format(mapping))
        self.mapping = mapping

    def __bool__(self) -> bool:
        return bool(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterable[KT]:
        for key in self.mapping:
            yield key

    def __getitem__(self, item: KT) -> VT:
        return self.mapping[item] if isinstance(self.mapping, Mapping)\
            else self.mapping(item)

    def __setitem__(self, key: KT, value: VT) -> None:
        """
        Sets the mapped value to a specified key.

        Example
        -------
        >>> mapping = MappingOrCallable({1: 'a', 2: 'b'})
        >>> mapping[1] = 'X'
        >>> print(mapping)
        {1: 'X', 2: 'b'}
        """
        self.mapping[key] = value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MappingOrCallable):
            return self.mapping == other.mapping
        return self.mapping == other

    def __repr__(self):
        return repr(self.mapping)

    def then(self, other: Mapping[VT, V2T] | Callable[[VT], V2T]
             ) -> MappingOrCallable[KT, V2T]:
        """
        Returns the composition of the object with a dict or a Callable.

        Example
        -------
        >>> mapping = MappingOrCallable({1: 'a'})
        >>> assert mapping.then({'a': 1}) == mapping.then(len) == {1: 1}
        """
        other = other if isinstance(other, Mapping)\
            else MappingOrCallable(other)
        if isinstance(self.mapping, Mapping):
            return MappingOrCallable({
                key: other[self[key]] for key in self.mapping})
        return MappingOrCallable(lambda key: other[self[key]])


def identity(x: KT) -> KT:
    """
    The identity function, which every identity functor is built from so that
    it can be recognised, see :meth:`discopy.cat.Functor.then`.

    Example
    -------
    >>> assert identity(42) == 42
    """
    return x


def get_origin(typ):
    """
    Get origin of a parameterized generic type, leaving a union type
    untouched so that ``isinstance`` can be called on it directly.

    Example
    -------
    >>> assert get_origin(tuple[int, str]) is tuple
    >>> assert get_origin(int | None) == (int | None)
    """
    origin = getattr(typ, "__origin__", typ)
    return typ if origin is Union else origin


@lru_cache(maxsize=None)
def keywords(function: Callable) -> frozenset[str] | None:
    """
    The names a function takes by keyword, ``None`` when it takes any, i.e.
    when it has ``**kwargs``, read from its signature once per function.

    Example
    -------
    >>> assert keywords(lambda x, /, y, *, z: x) == {"y", "z"}
    >>> assert keywords(lambda x, **kwargs: x) is None
    """
    parameters = inspect.signature(function).parameters.values()
    if any(param.kind == param.VAR_KEYWORD for param in parameters):
        return None
    return frozenset(param.name for param in parameters if param.kind in (
        param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY))


def rebuild_named_generic(func, origin, values, args):
    """
    Unpickle a member of a subscripted :class:`NamedGeneric` by calling the
    reconstructor ``func`` of its pickle on the subscripted class itself,
    which is found by subscripting its ``origin`` again with its ``values``,
    each decoded from its :class:`Subscript`.
    """
    return func(origin[tuple(map(Subscript.decode, values))], *args)


class Subscript:
    """
    A class subscripted by :class:`NamedGeneric`, as its origin and its
    values. Such a class is created on the fly, so it cannot be pickled by
    name: a value that is one, e.g. the ``Cyclic[2]`` of
    ``Product[Cyclic[2], Symmetric[3]]``, is pickled as its subscript.

    Parameters:
        origin : The class subscripted.
        values : The values it is subscripted by.

    Example
    -------
    >>> from discopy.matrix import Matrix
    >>> subscript = Subscript.encode(Matrix[int])
    >>> subscript.origin is Matrix and subscript.values == (int, )
    True
    >>> assert Subscript.decode(subscript) is Matrix[int]
    """
    def __init__(self, origin: type, values: tuple):
        self.origin, self.values = origin, values

    @classmethod
    def encode(cls, value):
        """ The subscript of a subscripted class, else the value itself. """
        if isinstance(value, type) and "__is_named_generic__" in vars(value):
            return cls(value.__origin__,
                       tuple(map(cls.encode, value.__values__)))
        return value

    @staticmethod
    def decode(value):
        """ The class of a subscript, else the value itself. """
        if isinstance(value, Subscript):
            return value.origin[tuple(map(Subscript.decode, value.values))]
        return value


class NamedGeneric:
    """
    A ``NamedGeneric`` is a class whose type parameters are attached by name
    to its members.

    Note
    ----
    In a standard ``Generic`` class, the type parameter disappears when the
    member of the class is instantiated, e.g.

    >>> assert list[int]([1, 2, 3])\\
    ...     == list[float]([1, 2, 3])\\
    ...     == [1, 2, 3]

    In a ``NamedGeneric``, the type parameter is attached to the members of the
    class so that we have access to it.

    Example
    -------

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class L[dtype](NamedGeneric):
    ...     inside: list
    >>> assert L[int]([1, 2, 3]).dtype == int
    >>> assert L[int]([1, 2, 3]) != L[float]([1, 2, 3])

    A subscript of a class that is its own :func:`factory`, i.e. the arrows
    of a category, is its own factory in turn, so that the structural arrows
    it builds through ``cls.ar`` carry the parameter.

    >>> from discopy.abc import Category
    >>> @factory
    ... class C[dtype](Category, NamedGeneric):
    ...     pass
    >>> assert C[int].factory is C[int] is C[int].ar

    A DisCoPy class is named by its module, since every level of the
    hierarchy has a ``Diagram`` of its own:

    >>> from discopy import frobenius
    >>> frobenius.Hypergraph.__name__
    'Hypergraph[frobenius.Diagram]'
    """
    if TYPE_CHECKING:
        #: The names of the type parameters in DisCoPy, declared so that
        #: reading them typechecks: subscripting attaches their values.
        dtype: ClassVar[Any]
        category: ClassVar[Any]
        base: ClassVar[Any]
        natural: ClassVar[Any]
        algebra: ClassVar[Any]
        generator_factory: ClassVar[Any]

    _cache = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for param in cls.__type_params__:
            if not hasattr(cls, param.__name__):
                setattr(cls, param.__name__, None)

    def __class_getitem__(cls, values):
        """
        Subscripting with values builds a subclass carrying them, while
        subscripting with the type parameters of the class syntax, e.g.
        ``class Box[dtype](Diagram[dtype])``, declares a plain subclass.
        """
        values = values if isinstance(values, tuple) else (values,)
        if any(isinstance(value, TypeVar) and value.__infer_variance__
               for value in values):
            return super().__class_getitem__(
                values[0] if len(values) == 1 else values)
        if hasattr(cls, "__is_named_generic__"):
            cls = cls.__bases__[0]
        attributes = next((
            [param.__name__ for param in c.__type_params__]
            for c in cls.__mro__ if c.__type_params__), None)
        if attributes is None:
            raise TypeError(messages.NO_TYPE_PARAMETER.format(cls))
        cls_values = tuple(getattr(cls, attr, None) for attr in attributes)
        if cls not in NamedGeneric._cache:
            NamedGeneric._cache[cls] = {cls_values: cls}
        if values not in NamedGeneric._cache[cls]:
            origin = get_origin(cls)

            class C(origin):
                __is_named_generic__ = True

                def __reduce__(self):
                    """
                    Pickle a member of the subscripted class by its
                    origin and values, since a class created inside
                    a function cannot be found by name, see `how can
                    I pickle a dynamically created nested class
                    <https://stackoverflow.com/questions/1947904>`_.
                    The class is rebuilt before the state is set,
                    so that unpickling needs no ``__setstate__``,
                    which the subscripted class does not inherit.

                    >>> import pickle
                    >>> from discopy.tensor import Tensor, Dim
                    >>> t = Tensor[int]([1, 0, 0, 1], Dim(2), Dim(2))
                    >>> u = pickle.loads(pickle.dumps(t))
                    >>> assert type(u) is Tensor[int] and u == t
                    """
                    func, args, *rest = super().__reduce__()
                    if not args or args[0] is not C:
                        return (func, args, *rest)
                    return (
                        rebuild_named_generic,
                        (func, origin, tuple(map(Subscript.encode, values)),
                         args[1:]), *rest)

            C.__module__ = origin.__module__
            names = [
                factory_name(v) if is_discopy_class(v)
                else getattr(v, "__name__", str(v)) for v in values]
            C.__name__ = C.__qualname__ = origin.__name__\
                + f"[{', '.join(names)}]"
            C.__origin__, C.__values__ = cls, values
            for attr, value in zip(attributes, values):
                setattr(C, attr, value)
            if getattr(origin, "factory", None) is origin:
                C.factory = C
            NamedGeneric._cache[cls][values] = C
        return NamedGeneric._cache[cls][values]

    def __setstate__(self, state):
        if "__class_getitem__values__" in state:
            state = dict(state)
            self.__class__ = self.__class__[tuple(map(
                Subscript.decode, state.pop("__class_getitem__values__")))]
        setstate = getattr(super(), "__setstate__", None)
        if setstate is None:
            self.__dict__.update(state)
        else:
            setstate(state)


def product(xs: Sequence, unit=1):
    """
    The left-fold product of a ``unit`` with list of ``xs``.

    Example
    -------
    >>> assert product([1, 2, 3]) == 6
    >>> assert product([1, 2, 3], unit=[42]) == 6 * [42]
    """
    return unit if not xs else product(xs[1:], unit * xs[0])


def deprecated_alias(module_name: str, aliases: dict[str, str]):
    """
    The module-level ``__getattr__`` of a module with one or more classes
    that were renamed, returning each new class with a
    :class:`DeprecationWarning`.

    Parameters:
        module_name : The ``__name__`` of the module deprecating names.
        aliases : A mapping from each deprecated name to its new name, in
            the same module, or to its new dotted path when it moved.

    Example
    -------
    >>> import warnings
    >>> from discopy import rigid, markov, comarkov
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     assert rigid.PRO is rigid.Nat
    ...     assert markov.Merge is comarkov.Merge
    >>> print(w[0].message)
    discopy.rigid.PRO is deprecated, use discopy.rigid.Nat instead.
    >>> print(w[1].message)
    discopy.markov.Merge is deprecated, use discopy.comarkov.Merge instead.
    """
    def __getattr__(name):
        if name in aliases:
            import warnings
            from importlib import import_module
            new_module, _, new_name = aliases[name].rpartition('.')
            new_module = new_module or module_name
            warnings.warn(
                f"{module_name}.{name} is deprecated, "
                f"use {new_module}.{new_name} instead.",
                DeprecationWarning, stacklevel=2)
            return getattr(import_module(new_module), new_name)
        raise AttributeError(
            f"module {module_name!r} has no attribute {name!r}")
    return __getattr__


def is_discopy_class(value) -> bool:
    """
    Whether a value is a class defined in DisCoPy, which
    :class:`NamedGeneric` names by its :func:`factory_name`, e.g.
    ``hypergraph.Hypergraph[frobenius.Diagram]``, since every level of the
    hierarchy has a ``Diagram`` of its own.

    >>> from discopy.frobenius import Diagram
    >>> assert is_discopy_class(Diagram) and not is_discopy_class(int)
    """
    return isinstance(value, type) and (
        value.__module__ == "discopy"
        or value.__module__.startswith("discopy."))


def factory_name(cls: type) -> str:
    """
    Returns a string describing a DisCoPy class.

    Example
    -------
    >>> from discopy.grammar.pregroup import Word
    >>> assert factory_name(Word) == "grammar.pregroup.Word"
    >>> assert factory_name(int | None) == "int | None"
    """
    if not isinstance(cls, type):
        return str(cls).removeprefix('discopy.')
    module = cls.__module__.removeprefix('discopy.')
    return f"{module}.{cls.__name__}".removeprefix('builtins.')


def from_tree(tree: dict):
    """
    Import DisCoPy and decode a serialised object.

    Parameters:
        tree : The serialisation of a DisCoPy object.

    Example
    -------
    >>> tree = {'factory': 'cat.Arrow',
    ...         'inside': [   {   'factory': 'cat.Box',
    ...                           'name': 'f',
    ...                           'dom': {'factory': 'cat.Ob', 'name': 'x'},
    ...                           'cod': {'factory': 'cat.Ob', 'name': 'y'},
    ...                           'data': 42},
    ...                       {   'factory': 'cat.Box',
    ...                           'name': 'f',
    ...                           'dom': {'factory': 'cat.Ob', 'name': 'y'},
    ...                           'cod': {'factory': 'cat.Ob', 'name': 'x'},
    ...                           'is_dagger': True,
    ...                           'data': 42}],
    ...         'dom': {'factory': 'cat.Ob', 'name': 'x'},
    ...         'cod': {'factory': 'cat.Ob', 'name': 'x'}}

    >>> from discopy.cat import Box
    >>> f = Box('f', 'x', 'y', data=42)
    >>> assert from_tree(tree) == f >> f[::-1]
    """
    return resolve_factory(tree['factory']).from_tree(tree)


def resolve_factory(name: str) -> type:
    """
    The DisCoPy class with a given :func:`factory_name`, the inverse of it:
    a subscripted class is resolved by its origin and its parameter, which
    may be a type of ``builtins`` or of ``numpy``, the two places a data type
    parameterising a category comes from.

    >>> from discopy.markov import Diagram
    >>> assert resolve_factory(factory_name(Diagram)) is Diagram
    >>> from numpy import float64
    >>> from discopy.tensor import Diagram
    >>> assert resolve_factory("tensor.Diagram[complex]") is Diagram[complex]
    >>> assert resolve_factory("tensor.Diagram[float64]") is Diagram[float64]

    Any other name raises, e.g. that of an object a module of DisCoPy
    imports, so that a file names no function for :func:`loads` to reach.

    >>> resolve_factory("utils.json.loads")
    Traceback (most recent call last):
    ...
    AttributeError: No class of DisCoPy is named 'utils.json.loads'.
    """
    if name.endswith(']'):
        origin, parameter = name[:-1].split('[', 1)
        return resolve_factory(origin)[
            resolve_factory(parameter)]  # ty: ignore[not-subscriptable]
    *modules, factory = name.removeprefix('discopy.').split('.')
    import discopy  # pylint: disable=import-outside-toplevel
    if not modules and factory not in vars(discopy):
        import numpy
        for module in (builtins, numpy):
            if isinstance(vars(module).get(factory), type):
                return vars(module)[factory]
        raise AttributeError(messages.NO_SUCH_TYPE.format(factory))
    module = sys.modules.get(".".join(["discopy", *modules]))
    cls = getattr(module, factory, None)
    if not is_discopy_class(cls):
        raise AttributeError(messages.NO_SUCH_CLASS.format(name))
    return cls


SYMPY_CONSTANTS = (
    "Zero", "One", "NegativeOne", "Half", "Pi", "Exp1", "ImaginaryUnit",
    "Infinity", "NegativeInfinity", "ComplexInfinity", "NaN", "EulerGamma",
    "GoldenRatio", "Catalan")
""" The sympy singletons a tree names, read back from ``sympy.S``. """

SYMPY_FUNCTIONS = (
    "Add", "Mul", "Pow", "Mod", "Min", "Max", "Abs", "sign", "floor",
    "ceiling", "re", "im", "arg", "conjugate", "exp", "log", "sin", "cos",
    "tan", "cot", "sec", "csc", "asin", "acos", "atan", "acot", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh")
""" The sympy classes a tree applies to its ``args``, read from ``sympy``. """


def encode(tree: Any) -> Any:
    """
    A tree as JSON can write it, with a node tagged by its factory for each
    value that JSON has no notation for, or one it would read back as
    something else: a tuple, which it writes as a list, a complex number, see
    :func:`encode_complex`, a fraction, see :func:`encode_fraction`, a numpy
    array or scalar, see :func:`encode_numpy`,
    and a sympy expression, see :func:`encode_sympy`. Every other value is
    left as it is, so a tree without tuples keeps the bytes JSON gives it, as
    is one that these refuse, for the ``default`` of ``json.dumps``.
    This is what :func:`dumps` writes and :func:`decode` reads back.

    Example
    -------
    >>> encode({'name': 'f', 'data': (1, 2j)})
    {'name': 'f', 'data': {'factory': 'tuple', 'inside': \
[1, {'factory': 'complex', 're': 0.0, 'im': 2.0}]}}
    >>> encode({'data': {1, 2}})
    {'data': {1, 2}}
    """
    if isinstance(tree, dict):
        return {key: encode(value) for key, value in tree.items()}
    if isinstance(tree, list | tuple):
        inside = list(map(encode, tree))
        return inside if isinstance(tree, list)\
            else {'factory': factory_name(tuple), 'inside': inside}
    if tree is None or isinstance(tree, bool | int | float | str):
        return tree
    if isinstance(tree, complex):
        return encode_complex(tree)
    if isinstance(tree, Fraction):
        return encode_fraction(tree)
    package = (type(tree).__module__ or "").partition('.')[0]
    encoder = {"numpy": encode_numpy, "sympy": encode_sympy}.get(package)
    try:
        return tree if encoder is None else encoder(tree)
    except TypeError:
        return tree


def decode(tree: dict) -> Any:
    """
    Read back a node that :func:`encode` wrote, leaving any other alone.

    This is the ``object_hook`` of :func:`loads`, i.e. :mod:`json` calls it on
    every object it decodes, innermost first, so it answers a node that it did
    not write with that node itself. A box whose ``data`` is a dictionary of
    its own is left alone by the factory tag, which a tree reserves: a node is
    read back only when it has exactly the keys that :func:`encode` writes,
    holding values of the types it writes.

    Example
    -------
    >>> assert decode({'factory': 'tuple', 'inside': [1, 2]}) == (1, 2)
    >>> tree = {'factory': 'tuple', 'inside': [1, 2], 'name': 'x'}
    >>> assert decode(tree) == tree
    """
    name = tree.get('factory')
    if name == factory_name(tuple) and tree.keys() == {'factory', 'inside'}\
            and isinstance(tree['inside'], list):
        return tuple(tree['inside'])
    if isinstance(name, str) and name.startswith("numpy."):
        return decode_numpy(tree)
    if isinstance(name, str) and name.startswith("sympy."):
        return decode_sympy(tree)
    if name == factory_name(Fraction):
        return decode_fraction(tree)
    return decode_complex(tree)


def encode_numpy(value: "numpy.ndarray | numpy.generic") -> dict:
    """
    A numpy array as a node with its ``dtype``, its ``shape`` and its
    entries in order, i.e. ``flat``, or a numpy scalar as a node with its
    ``item``, i.e. its value as a Python scalar, see :func:`encode`. It
    raises on any value that :func:`decode_numpy` does not read back.

    Example
    -------
    >>> import numpy
    >>> encode_numpy(numpy.array([[1, 0], [0, 1]]))
    {'factory': 'numpy.ndarray', 'dtype': 'int64', 'shape': [2, 2], \
'flat': [1, 0, 0, 1]}
    >>> encode_numpy(numpy.float32(0.5))
    {'factory': 'numpy.float32', 'item': 0.5}
    >>> encode_numpy(numpy.str_('a'))
    Traceback (most recent call last):
    ...
    TypeError: Object of type str_ is not JSON serialisable.
    """
    import numpy
    tree = {'factory': factory_name(type(value))}
    if type(value) is numpy.ndarray:
        tree |= {'dtype': str(value.dtype), 'shape': list(value.shape),
                 'flat': value.reshape(-1).tolist()}
    elif isinstance(value, numpy.generic):
        tree['item'] = value.item()
    if decode_numpy(tree) is tree:
        raise TypeError(messages.NOT_JSON_SERIALISABLE.format(
            type(value).__name__))
    return encode(tree)


def decode_numpy(tree: dict) -> Any:
    """
    Read back what :func:`encode_numpy` wrote, leaving any other node alone:
    its scalar type is one that numpy names for booleans, integers, floats or
    complex numbers, and its entries are the Python scalars that type reads
    back as, in its range for an integer, so that an array allocates in
    proportion to the file and no string is parsed as a number. A shape
    that numpy refuses, e.g. one with a dimension too large, is left alone.

    Example
    -------
    >>> import numpy
    >>> array = numpy.arange(6).reshape(2, 3)
    >>> assert (decode_numpy(encode_numpy(array)) == array).all()
    >>> scalar = decode_numpy(encode_numpy(numpy.int8(-3)))
    >>> assert scalar == -3 and isinstance(scalar, numpy.int8)
    >>> for item in ('-3', 1000):
    ...     tree = {'factory': 'numpy.int8', 'item': item}
    ...     assert decode_numpy(tree) == tree
    """
    import numpy
    name, fields = tree['factory'].removeprefix("numpy."), tree.keys()
    if name == "ndarray" and fields == {'factory', 'dtype', 'shape', 'flat'}:
        dtype, entries, shape = tree['dtype'], tree['flat'], tree['shape']
    elif fields == {'factory', 'item'}:
        dtype, entries, shape = name, [tree['item']], None
    else:
        return tree
    scalar = numpy.sctypeDict.get(dtype) if isinstance(dtype, str) else None
    if scalar is None or numpy.dtype(scalar).kind not in "biufc"\
            or not isinstance(entries, list):
        return tree
    python = type(scalar(0).item())
    bounds = numpy.iinfo(scalar) if python is int else None
    if python not in (bool, int, float, complex)\
            or any(type(entry) is not python for entry in entries)\
            or bounds is not None and not all(
                bounds.min <= entry <= bounds.max for entry in entries):
        return tree
    if shape is None:
        return scalar(tree['item'])
    if isinstance(shape, list)\
            and all(type(n) is int and n >= 0 for n in shape)\
            and prod(shape) == len(entries):
        try:
            return numpy.array(entries, dtype=scalar).reshape(shape)
        except (ValueError, OverflowError):
            return tree
    return tree


def encode_sympy(expr: "sympy.Basic") -> dict:
    """
    A sympy expression as a tree of sympy classes: a symbol by its ``name``
    and the ``assumptions`` it was given, an integer by its numerator ``p``, a
    rational by ``p`` and its denominator ``q``, a float by the decimal
    ``num`` that reads back exactly at its ``precision``, a constant in
    :data:`SYMPY_CONSTANTS` by its name alone and an expression whose head
    is in :data:`SYMPY_FUNCTIONS` by its ``args``.

    Any other class raises, e.g. an undefined function named ``sin``, so
    that :func:`decode_sympy` can read back what this writes without
    ``sympy.sympify``, which calls :func:`eval` on strings, i.e.
    :func:`loads` runs no code from a file.

    Example
    -------
    >>> from sympy import Dummy, Symbol, pi  # doctest: +EXTRA
    >>> phi = Symbol('phi', real=True)
    >>> encode_sympy(phi / 2 + pi)
    {'factory': 'sympy.Add', 'args': [{'factory': 'sympy.Pi'}, \
{'factory': 'sympy.Mul', 'args': [{'factory': 'sympy.Half'}, \
{'factory': 'sympy.Symbol', 'name': 'phi', 'assumptions': {'real': True}}]}]}
    >>> encode_sympy(Dummy())
    Traceback (most recent call last):
    ...
    TypeError: Object of type Dummy is not JSON serialisable.
    >>> from sympy import Function
    >>> encode_sympy(phi + Function('sin')(phi))
    Traceback (most recent call last):
    ...
    TypeError: Object of type sin is not JSON serialisable.
    """
    import sympy
    name = type(expr).__name__
    tree = {'factory': f"sympy.{name}"}
    if name in SYMPY_CONSTANTS and expr is getattr(sympy.S, name):
        return tree
    if type(expr) is not getattr(sympy, name, None)\
            or name == "Symbol" and not expr.name.isidentifier():
        raise TypeError(messages.NOT_JSON_SERIALISABLE.format(name))
    if name == "Symbol":
        _, assumptions = expr.__getnewargs_ex__()
        tree['name'] = expr.name
        tree.update({'assumptions': assumptions} if assumptions else {})
    elif name in ("Integer", "Rational"):
        tree.update({'p': expr.p} | ({'q': expr.q} if expr.q != 1 else {}))
    elif name == "Float":
        from mpmath.libmp import repr_dps
        _, options = expr.__getnewargs_ex__()
        digits = repr_dps(options['precision'])
        tree['num'] = str(expr.evalf(digits)) if expr else "0." + "0" * digits
        tree['precision'] = options['precision']
    elif name in SYMPY_FUNCTIONS:
        tree['args'] = list(map(encode_sympy, expr.args))
    else:
        raise TypeError(messages.NOT_JSON_SERIALISABLE.format(name))
    return tree


def decode_sympy(tree: dict) -> Any:
    """
    Read back what :func:`encode_sympy` wrote, leaving any other node alone.

    Each class is called on fields of the types it is written with, e.g. a
    function on sympy expressions only, never on a string, which sympy would
    parse with :func:`eval`, and with ``evaluate=False``, so that it rebuilds
    the expression as it was written, without computing e.g. a large power.
    A float is read at a precision of at most four bits a digit, so that it
    allocates in proportion to the file. A node with exactly the fields
    :func:`encode_sympy` writes for its class is still left alone when sympy
    refuses them, e.g. a symbol with inconsistent assumptions or a function
    with the wrong number of ``args``.
    As the ``object_hook`` of :func:`loads` it reads the innermost nodes
    first, i.e. the ``args`` of a node come decoded.

    Example
    -------
    >>> from sympy import Symbol, sin  # doctest: +EXTRA
    >>> theta = decode_sympy({'factory': 'sympy.Symbol', 'name': 'theta'})
    >>> assert decode_sympy({'factory': 'sympy.sin', 'args': [theta]})\\
    ...     == sin(Symbol('theta'))
    >>> tree = {'factory': 'sympy.sin', 'args': ['__import__("os")']}
    >>> assert decode_sympy(tree) == tree
    >>> tree = {'factory': 'sympy.sin', 'args': [theta, theta]}
    >>> assert decode_sympy(tree) == tree
    """
    import sympy
    name, fields = tree['factory'].removeprefix("sympy."), tree.keys()
    assumptions = tree.get('assumptions', {})
    if name == "Symbol" and fields <= {'factory', 'name', 'assumptions'}\
            and isinstance(tree.get('name'), str)\
            and tree['name'].isidentifier()\
            and isinstance(assumptions, dict)\
            and all(isinstance(value, bool) for value in assumptions.values()):
        try:
            return sympy.Symbol(tree['name'], **assumptions)
        except (TypeError, ValueError):
            return tree
    if name == "Integer" and fields == {'factory', 'p'}\
            and type(tree['p']) is int:
        return sympy.Integer(tree['p'])
    if name == "Rational" and fields == {'factory', 'p', 'q'}\
            and type(tree['p']) is int and type(tree['q']) is int\
            and tree['q'] > 1:
        return sympy.Rational(tree['p'], tree['q'])
    if name == "Float" and fields == {'factory', 'num', 'precision'}\
            and isinstance(tree['num'], str)\
            and type(tree['precision']) is int\
            and tree['precision'] <= 4 * len(tree['num']):
        return sympy.Float(tree['num'], precision=tree['precision'])
    if name in SYMPY_CONSTANTS and fields == {'factory'}:
        return getattr(sympy.S, name)
    if name in SYMPY_FUNCTIONS and fields == {'factory', 'args'}\
            and isinstance(tree['args'], list)\
            and all(isinstance(arg, sympy.Basic) for arg in tree['args']):
        try:
            return getattr(sympy, name)(*tree['args'], evaluate=False)
        except (TypeError, ValueError):
            return tree
    return tree


def encode_complex(number: complex) -> dict:
    """
    A complex number as a node of a tree, since JSON has only reals.

    :math:`\\mathbb{C}` is :math:`\\mathbb{R}^2` as a real vector space, so a
    complex number travels as the pair of its real and imaginary parts, tagged
    with its factory like every other node, see :func:`encode`. It raises on
    anything else.

    Example
    -------
    >>> encode_complex(1 + 2j)
    {'factory': 'complex', 're': 1.0, 'im': 2.0}
    >>> encode_complex({1, 2})
    Traceback (most recent call last):
    ...
    TypeError: Object of type set is not JSON serialisable.
    """
    if not isinstance(number, complex):
        raise TypeError(messages.NOT_JSON_SERIALISABLE.format(
            type(number).__name__))
    return {'factory': factory_name(complex),
            're': number.real, 'im': number.imag}


def decode_complex(tree: dict) -> Any:
    """
    Read back what :func:`encode_complex` wrote, leaving other nodes alone,
    see :func:`decode`: the one dictionary read as a complex number is
    ``{'factory': 'complex', 're': x, 'im': y}``, with exactly those three
    keys, as :func:`encode_complex` writes it.

    Example
    -------
    >>> assert decode_complex(encode_complex(1 + 2j)) == 1 + 2j
    >>> tree = {'factory': 'cat.Ob', 'name': 'x'}
    >>> assert decode_complex(tree) == tree
    >>> assert decode_complex({'re': 1, 'im': 2}) == {'re': 1, 'im': 2}
    """
    if tree.keys() == {'factory', 're', 'im'} \
            and tree['factory'] == factory_name(complex):
        return complex(tree['re'], tree['im'])
    return tree


def encode_fraction(number: Fraction) -> dict:
    """
    A fraction as a node of a tree, since JSON has only integers and floats:
    its ``numerator`` and ``denominator`` in lowest terms, tagged with its
    factory like every other node, see :func:`encode`.

    Example
    -------
    >>> encode_fraction(Fraction(2, 8))
    {'factory': 'fractions.Fraction', 'numerator': 1, 'denominator': 4}
    """
    return {'factory': factory_name(Fraction),
            'numerator': number.numerator, 'denominator': number.denominator}


def decode_fraction(tree: dict) -> Any:
    """
    Read back what :func:`encode_fraction` wrote, leaving other nodes alone,
    see :func:`decode`: exactly its three keys, with an integer numerator
    and a positive integer denominator.

    Example
    -------
    >>> assert decode_fraction(encode_fraction(Fraction(-1, 4)))\\
    ...     == Fraction(-1, 4)
    >>> tree = {'factory': 'fractions.Fraction',
    ...         'numerator': 1, 'denominator': 0}
    >>> assert decode_fraction(tree) == tree
    """
    if tree.keys() == {'factory', 'numerator', 'denominator'}\
            and tree['factory'] == factory_name(Fraction)\
            and type(tree['numerator']) is int\
            and type(tree['denominator']) is int\
            and tree['denominator'] > 0:
        return Fraction(tree['numerator'], tree['denominator'])
    return tree


def dumps(obj, **kwargs):
    """
    Serialise a DisCoPy object as JSON.

    Parameters:
        obj : The DisCoPy object to serialise, or a list of them, which
            :func:`loads` reads back as a list, e.g. a corpus.
        kwargs : Passed to ``json.dumps``, e.g. a ``default`` or a ``cls``
            for the values that :func:`encode` leaves alone, on which it
            raises otherwise.

    The tree of ``obj`` goes through :func:`encode`, so that its data may hold
    tuples, complex numbers, numpy arrays and scalars and sympy expressions.

    Example
    -------
    >>> from discopy.cat import Box, Id
    >>> f = Box('f', 'x', 'y', data=42)
    >>> print(dumps(f[::-1] >> Id('x'), indent=4))
    {
        "factory": "cat.Arrow",
        "inside": [
            {
                "factory": "cat.Box",
                "name": "f",
                "dom": {
                    "factory": "cat.Ob",
                    "name": "y"
                },
                "cod": {
                    "factory": "cat.Ob",
                    "name": "x"
                },
                "is_dagger": true,
                "data": 42
            }
        ],
        "dom": {
            "factory": "cat.Ob",
            "name": "y"
        },
        "cod": {
            "factory": "cat.Ob",
            "name": "x"
        }
    }
    """
    if 'cls' not in kwargs:
        kwargs.setdefault('default', encode_complex)
    tree = [x.to_tree() for x in obj] if isinstance(obj, list)\
        else obj.to_tree()
    return json.dumps(encode(tree), **kwargs)


def loads(raw):
    """
    Loads a serialised DisCoPy object, decoding its nodes with
    :func:`decode`, which runs no code from ``raw``.

    Example
    -------
    >>> raw = '{"factory": "cat.Ob", "name": "x"}'
    >>> from discopy.cat import Ob
    >>> assert loads(raw) == Ob('x')
    >>> assert dumps(loads(raw)) == raw
    >>> assert loads(dumps(Ob('x'))) == Ob('x')
    >>> assert loads(dumps([Ob('x'), Ob('y')])) == [Ob('x'), Ob('y')]
    """
    obj = json.loads(raw, object_hook=decode)
    if isinstance(obj, list):
        return [from_tree(o) for o in obj]
    return from_tree(obj)


def rmap(func, data):
    """
    Apply :code:`func` recursively to :code:`data`, i.e. to the leaves of its
    mappings, iterables and numpy arrays, where a string is a leaf.

    Example
    -------
    >>> data = {'A': [0, 1, 2], 'B': ({'C': 3, 'D': [4, 5, 6]}, {7, 8, 9})}
    >>> rmap(lambda x: x + 1, data)
    {'A': [1, 2, 3], 'B': ({'C': 4, 'D': [5, 6, 7]}, {8, 9, 10})}
    >>> rmap(str.upper, ["ab", ("c", )])
    ['AB', ('C',)]
    >>> import numpy
    >>> rmap(lambda x: x / 2, numpy.array([[0, 1], [2, 3]]))
    array([[0. , 0.5],
           [1. , 1.5]])
    """
    import numpy
    if isinstance(data, Mapping):
        return {key: rmap(func, value) for key, value in data.items()}
    if isinstance(data, numpy.ndarray):
        return numpy.array(rmap(func, data.tolist())) if data.size\
            else data.copy()
    if isinstance(data, Iterable) and not isinstance(data, str):
        return type(data)([rmap(func, elem) for elem in data])
    return func(data)


def rsubs(data, *args, **kwargs):
    """
    Substitute recursively along nested data, see :func:`rmap`.

    Parameters:
        data : The data in which to substitute.
        args : A variable and an expression or the pair of them, or for
            multiple substitution, a list or a mapping of them, as in
            :code:`sympy.subs`, or the pairs themselves.
        kwargs : Passed to :code:`sympy.lambdify`.

    Example
    -------
    >>> from sympy.abc import phi, psi  # doctest: +EXTRA
    >>> data = {"name": "phi", "phases": [phi, phi + psi]}
    >>> rsubs(data, phi, 1)
    {'name': 'phi', 'phases': [1, psi + 1]}
    >>> rsubs(data, [(phi, 1), (psi, 2)])
    {'name': 'phi', 'phases': [1, 3]}
    >>> assert rsubs(data, [(phi, 1), (psi, 2)])\\
    ...     == rsubs(data, {phi: 1, psi: 2}) == rsubs(data, (phi, 1), (psi, 2))
    >>> assert rsubs(data, (phi, 1)) == rsubs(data, phi, 1)

    A leaf that keeps a free symbol is substituted symbolically:

    >>> from sympy import sin
    >>> rsubs(sin(phi) + sin(psi), phi, 1)
    sin(psi) + sin(1)

    A key that is not a symbol is substituted as in :code:`sympy.subs`:

    >>> rsubs(data, phi + psi, 0)
    {'name': 'phi', 'phases': [phi, 0]}
    """
    from sympy import Basic, MatrixBase, Symbol, lambdify
    pairs = args[0] if len(args) == 1 else args
    if not isinstance(pairs, Mapping):
        pairs = tuple(pairs)
        if pairs and not isinstance(pairs[0], Iterable):
            pairs = (pairs, )
    substitution = dict(pairs)
    compound = not all(isinstance(key, Symbol) for key in substitution)

    def substitute(leaf):
        if compound and isinstance(leaf, (Basic, MatrixBase)):
            return leaf.subs(substitution, simultaneous=True)
        symbols = tuple(getattr(leaf, "free_symbols", ()))
        if not substitution.keys() & set(symbols):
            return leaf
        values = tuple(substitution.get(x, x) for x in symbols)
        if any(isinstance(value, Basic) for value in values):
            return leaf.subs(dict(zip(symbols, values)), simultaneous=True)
        return lambdify(symbols, leaf, **kwargs)(*values)
    return rmap(substitute, data)


def load_corpus(url):
    """ Load a corpus hosted at a given ``url``. """
    import urllib.request as urllib
    import zipfile

    fd, _ = urllib.urlretrieve(url)
    zip_file = zipfile.ZipFile(fd, 'r')
    first_file = zip_file.namelist()[0]
    with zip_file.open(first_file) as f:
        return loads(f.read())


def assert_isinstance(object_, cls: type | tuple[type, ...]):
    """ Raise ``TypeError`` if ``object`` is not instance of ``cls``. """
    classes = cls if isinstance(cls, tuple) else (cls, )
    for cls in classes:
        if isinstance(object_, get_origin(cls)):
            return
    cls_name = ' | '.join(map(factory_name, classes))
    raise TypeError(messages.TYPE_ERROR.format(
        cls_name, factory_name(type(object_))))


def assert_ismember(object_, cls: type):
    """
    Raise ``TypeError`` unless ``object_`` is an instance of exactly ``cls``:
    unlike :func:`assert_isinstance`, the parameters of a subscript count,
    e.g. ``Cyclic[3](1)`` is not an element of ``Cyclic[2]`` and a channel of
    the maybe monad is not a channel of the powerset monad.

    >>> from discopy.group import Cyclic
    >>> assert_ismember(Cyclic[3](1), Cyclic[2])
    Traceback (most recent call last):
        ...
    TypeError: Expected group.Cyclic[2], got group.Cyclic[3] instead.
    """
    if type(object_) is not cls:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(messages.TYPE_ERROR.format(
            factory_name(cls), factory_name(type(object_))))


def unbiased(binary_method):
    """
    Turn a biased method with signature (self, other) to an unbiased one, i.e.
    with signature (self, *others), see the `nLab`_.

    One argument is the base case, more are folded by calling the method by
    its name on the running result, so that a subclass overriding the binary
    case has its own folded rather than the one decorated. An operand passed
    by keyword, e.g. ``f.then(other=g)``, goes to the binary method as it is,
    rather than being dropped as if there were none.

    .. _nLab: https://ncatlab.org/nlab/show/biased+definition
    """
    @wraps(binary_method)
    def method(self, *others, **params):
        if len(others) == 1 or not others and params:
            return binary_method(self, *others, **params)
        result = self
        for other in others:
            result = getattr(result, binary_method.__name__)(other, **params)
        return result
    return method


def inductive(induction_step):
    """ Turn a method with no input () to one with input (n_steps=1). """
    @wraps(induction_step)
    def method(self, n_steps=1):
        assert_isinstance(n_steps, int)
        if n_steps < 0:
            raise ValueError(messages.NEGATIVE_STEPS.format(n_steps))
        if n_steps == 0:
            return self
        return method(induction_step(self), n_steps - 1)
    return method


Pushout = tuple[dict[int, int], dict[int, int]]


def pushout(
        left: int, right: int,
        left_boundary: Collection[int], right_boundary: Collection[int]) \
        -> Pushout:
    """
    Computes the pushout of two finite mappings using connected components.

    Parameters:
        left : The size of the left set.
        right : The size of the right set.
        left_boundary : The mapping from boundary to left.
        right_boundary : The mapping from boundary to right.

    Examples
    --------
    >>> assert pushout(2, 3, [1], [0]) == ({0: 0, 1: 1}, {0: 1, 1: 2, 2: 3})
    """
    if len(left_boundary) != len(right_boundary):
        raise ValueError(messages.UNEQUAL_BOUNDARIES.format(
            left_boundary, right_boundary))
    components, left_pushout, right_pushout = set(), {}, {}
    left_proper = sorted(set(range(left)) - set(left_boundary))
    left_pushout.update({j: i for i, j in enumerate(left_proper)})
    graph = Graph([
        (("middle", i), ("left", j)) for i, j in enumerate(left_boundary)] + [
        (("middle", i), ("right", j)) for i, j in enumerate(right_boundary)])
    for i, component in enumerate(connected_components(graph)):
        components.add(i)
        for case, j in component:
            if case == "left":
                left_pushout[j] = len(left_proper) + i
            if case == "right":
                right_pushout[j] = len(left_proper) + i
    right_proper = set(range(right)) - set(right_boundary)
    right_pushout.update({
        j: len(left_proper) + len(components) + i
        for i, j in enumerate(right_proper)})
    return left_pushout, right_pushout


def set_partitions(items: list) -> Iterator[list[list]]:
    """
    The partitions of a list into non-empty blocks, all singletons first
    and one block last.

    >>> for blocks in set_partitions([0, 1, 2]): print(blocks)
    [[0], [1], [2]]
    [[0, 1], [2]]
    [[1], [0, 2]]
    [[0], [1, 2]]
    [[0, 1, 2]]
    """
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for blocks in set_partitions(rest):
        yield [[first]] + blocks
        for i, block in enumerate(blocks):
            yield blocks[:i] + [[first] + block] + blocks[i + 1:]


class BinaryBoxConstructor:
    """
    Box constructor with attributes ``left`` and ``right`` as input.

    Parameters:
        left : Some attribute on the left.
        right : Some attribute on the right.
    """
    def __init__(self, left, right):
        self.left, self.right = left, right

    def __setstate__(self, state):
        if "_name" in state:
            state["_name"] = type(self).__name__ + (
                              f"({state['right']}, {state['left']})"
                              if state.get("_is_dagger", False) else
                              f"({state['left']}, {state['right']})"
            )
        super().__setstate__(state)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({repr(self.left)}, {repr(self.right)})"

    def to_tree(self) -> dict:
        """ Serialise a binary box constructor. """
        left, right = self.left.to_tree(), self.right.to_tree()
        return dict(factory=factory_name(type(self)), left=left, right=right)

    @classmethod
    def from_tree(cls, tree: dict) -> BinaryBoxConstructor:
        """ Decode a serialised binary box constructor. """
        return cls(*map(from_tree, (tree['left'], tree['right'])))


@lru_cache(maxsize=1024)
def text_width(text: str, fontsize=12, points_per_inch=72., grid=16):
    """ The width of a text label in drawing units, i.e. inches.

    Measured from the actual glyph outlines with matplotlib's text layout,
    using the fixed font bundled with matplotlib so that different systems
    agree, at a given `fontsize` and `points_per_inch` conversion, then
    rounded up to the next `1 / grid` of an inch so that any remaining
    sub-pixel differences in font metrics across environments cannot change
    the layout of a diagram.
    """
    if not text:
        return 0
    width = TextPath(
        (0, 0), text, prop=_DEFAULT_FONT, size=fontsize,
        usetex=False).get_extents().width
    return ceil(width / points_per_inch * grid) / grid


def tuplify(stuff: Any, n: int | None = None) -> tuple:
    """
    Turns anything into a tuple, do nothing if it is already.

    Parameters:
        stuff : The stuff to turn into a tuple.
        n : The number of values ``stuff`` stands for, if known, so that a
            single value is wrapped whatever it is, a tuple included.

    Example
    -------
    >>> assert tuplify(1) == (1, ) and tuplify((1, 2)) == (1, 2)
    >>> assert tuplify((1, 2), n=1) == ((1, 2), )
    """
    return stuff if n != 1 and isinstance(stuff, tuple) else (stuff, )


def untuplify(stuff: tuple | list) -> Any:
    """
    Takes the element out of a tuple if it has length 1, otherwise do nothing.

    Parameters:
        stuff : The tuple out of which to take the element.

    Important
    ---------
    This is the inverse of :func:`tuplify`, except on tuples of length 1.
    """
    return stuff[0] if len(stuff) == 1 else stuff


def factory(cls):
    """
    Allows the identity and composition of an :class:`Arrow` subclass to
    remain within the subclass, by setting ``cls.factory = cls``.

    Parameters:
        cls : Some subclass of :class:`Arrow`.

    Note
    ----
    The factory method pattern (`FMP`_) is used all over DisCoPy. For
    backward compatibility, the factory is also available as the class
    property ``cls.ar``, see :class:`discopy.abc.Category`. A subscript of
    a factory, e.g. ``Tensor[complex]``, is its own factory, see
    :class:`NamedGeneric`.

    .. _FMP: https://en.wikipedia.org/wiki/Factory_method_pattern

    Example
    -------
    Let's create :code:`Circuit` as a subclass of :class:`Arrow` with an
    :class:`Ob` subclass :code:`Qubit` as domain and codomain.

    >>> from discopy.cat import Ob, Arrow, Box
    >>> class Qubit(Ob):
    ...     pass
    >>> @factory
    ... class Circuit(Arrow):
    ...     ob = Qubit

    The :code:`Circuit` subclass itself has a subclass :code:`Gate` as boxes.

    >>> class Gate(Box, Circuit):
    ...     pass

    The identity and composition of :code:`Circuit` is again a :code:`Circuit`.

    >>> X = Gate('X', Qubit(), Qubit())
    >>> assert isinstance(X >> X, Circuit)
    >>> assert isinstance(Circuit.id(), Circuit)
    >>> assert isinstance(Circuit.id().dom, Qubit)
    >>> assert Circuit.factory is Circuit.ar is Circuit
    """
    cls.factory = cls
    return cls


class AxiomError(Exception):
    """ The gods of category theory are not happy. """


def assert_iscomposable(left: Category, right: Category):
    """
    Raise :class:`AxiomError` if two objects are not composable,
    i.e. the domain of ``other`` is not the codomain of ``self``.

    Parameters:
        left : A composable object.
        right : Another composable object.
    """
    if not left.is_composable(right):
        raise AxiomError(messages.NOT_COMPOSABLE.format(
            left, right, left.cod, right.dom))


def assert_isparallel(left: Category, right: Category):
    """
    Raise :class:`AxiomError` if two composable objects do not have the
    same domain and codomain.

    Parameters:
        left : A composable object.
        right : Another composable object.
    """
    if not left.is_parallel(right):
        raise AxiomError(messages.NOT_PARALLEL.format(left, right))


def assert_isatomic(typ: Ty, cls: type | None = None):
    """ Raise :class:`AxiomError` if a type does not have length one. """
    cls = cls or type(typ)
    assert_isinstance(typ, cls)
    if not typ.is_atomic:
        raise ValueError(messages.NOT_ATOMIC.format(
            factory_name(cls), len(typ)))


def assert_istraceable(arg, n=1, left=False):
    """ Raise :class:`AxiomError` if a diagram is not traceable. """
    if n < 0 or n > min(len(arg.dom), len(arg.cod)):
        raise ValueError(messages.NOT_SPLITTABLE.format(n, arg))
    traced_dom, traced_cod = (arg.dom[:n], arg.cod[:n]) if left\
        else (arg.dom[len(arg.dom) - n:], arg.cod[len(arg.cod) - n:])
    if traced_dom != traced_cod:
        raise AxiomError(
            messages.NOT_TRACEABLE.format(traced_dom, traced_cod))


class classproperty(object):
    """ Adapted from https://stackoverflow.com/a/5192374/18783670 """
    def __init__(self, f):
        self.f = f

    def __get__(self, _, x):
        return self.f(x)


class Node:
    """ Node in a :class:`networkx.Graph`, can hold arbitrary data. """
    def __init__(self, kind, **data):
        self.kind, self.data = kind, data
        for key, value in data.items():
            setattr(self, key, value)
        self.__hash = None

    def __eq__(self, other):
        return isinstance(other, Node)\
            and (self.kind, self.data) == (other.kind, other.data)

    def __repr__(self):
        return f"""Node({repr(self.kind)}, {", ".join(
            f"{key}={value}" for key, value in sorted(self.data.items()))})"""

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash(repr(self))
        return self.__hash

    @property
    def index(self) -> tuple[int, int]:
        """ The position of the node in a diagram: its box, then its wire. """
        return self.data.get("j", -1), self.data.get("i", -1)

    def shift_i(self, i):
        return Node(self.kind, **dict(self.data, i=self.i + i))

    def shift_j(self, j):
        return Node(self.kind, **dict(self.data, j=self.j + j))


class Point(NamedTuple):
    """ A point is a pair of floats for the x and y coordinates. """
    x: float
    y: float

    def shift(self, x=0, y=0):
        return Point(self.x + x, self.y + y)


class RichDisplay:
    """
    Mixin implementing IPython's rich display protocol, see
    https://ipython.readthedocs.io/en/stable/config/integrating.html

    Any DisCoPy object with a :meth:`to_drawing` method, e.g. a
    :class:`discopy.monoidal.Diagram`, a :class:`discopy.drawing.Drawing`
    or an :class:`discopy.drawing.Equation`, is displayed as an SVG image
    (with a PNG fallback in the mimebundle) when it is the output of a cell
    in Jupyter, marimo and other frontends that support the protocol.
    :meth:`to_widget` shows it as a widget instead, on demand.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> f = Box('f', Ty('x'), Ty('y'))
    >>> svg, png = f._repr_svg_(), f.to_png()
    >>> assert svg.startswith('<?xml') and '</svg>\\n' in svg
    >>> assert png.startswith(b'\\x89PNG')
    >>> assert f._repr_mimebundle_(include=['image/svg+xml']) == {
    ...     'image/svg+xml': svg}
    >>> assert f._repr_mimebundle_(include=['image/png']) == {
    ...     'image/png': png}
    >>> assert f._repr_mimebundle_(include=['text/html']) == {}
    """
    def to_svg(self, **params) -> str:
        """ Draw as a standalone SVG string. """
        from io import StringIO
        buffer = StringIO()
        self.to_drawing().draw(
            path=buffer, format="svg", show=False, **params)
        return buffer.getvalue()

    def to_png(self, **params) -> bytes:
        """ Draw as PNG bytes. """
        from io import BytesIO
        buffer = BytesIO()
        self.to_drawing().draw(
            path=buffer, format="png", show=False, **params)
        return buffer.getvalue()

    def _repr_svg_(self) -> str:
        return self.to_svg()

    def to_widget(self, **params):
        """
        Draw as a :class:`discopy.drawing.widget.DiagramWidget`, which needs
        `anywidget <https://anywidget.dev>`_.

        Parameters:
            params : Passed to :meth:`to_svg`.
        """
        from discopy.drawing.widget import DiagramWidget
        return DiagramWidget(svg=self.to_svg(**params))

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        data = {"image/svg+xml": self.to_svg, "image/png": self.to_png}
        return {mimetype: draw() for mimetype, draw in data.items()
                if (include is None or mimetype in include)
                and (exclude is None or mimetype not in exclude)}
