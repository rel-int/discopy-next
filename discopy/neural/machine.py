# -*- coding: utf-8 -*-

"""
Recurrent machines and their coalgebras, the dual of the algebras of
:mod:`discopy.neural.datatype`, in *Categorical Deep Learning*
(arXiv:2402.15332).

A coalgebra :math:`S \\to F(S)` of :math:`F(S) = (I \\to O \\times S)`
(Mealy), :math:`O \\times (I \\to S)` (Moore) or :math:`O \\times S` (unfold)
is a cell from an input and the delayed state to an output and the state, a
:class:`Machine`. Its unfold is the feedback of the cell, the unique map into
the final coalgebra without parameters. With parameters, the tied map is the
homomorphism whose 2-cell is the copy of the parameters (App. J), a choice and
not forced. The semantics are the monoidal streams of :mod:`discopy.stream`:
the first steps are a finite diagram with one box for each time step and one
initial state as input. A :class:`Coalgebra` is a functor out of them, sending
every time step of a box to one cell and the initial state to one ``start``,
sampled once and copied by the cell, and the unrolled image ties the cells by
a copy of the parameter.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Machine
    Coalgebra
    ToStream

Note
----
This module imports no tensor framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import digits
from typing import Any

from discopy import feedback, markov, stream
from discopy.neural.datatype import Interpretation, check_param
from discopy.utils import assert_isinstance


class ToStream(markov.Functor):
    """
    The functor from diagrams to monoidal streams, sending an atom :math:`x`
    to the sequence of types ``x0, x1, ...`` and every box to the sequence of
    boxes of the same name and time step. A delayed atom is sent to the delay
    of the sequence, preceded by the singleton of the atom itself: at the
    first time step, where the delay is empty, this is the initial state.

    Example
    -------
    >>> x = markov.Ty("x")
    >>> image = ToStream()(x @ x.delay())
    >>> print(image.now, "|", image.later.now)
    x0 @ x | x1 @ x0
    """
    def __init__(self):
        super().__init__(
            ob_map=self.atom, ar_map=self.box,
            cod=stream.Stream[markov.Diagram])

    @staticmethod
    def atom(atom: markov.Ty) -> stream.Ty:
        """ The sequence ``x0, x1, ...`` of an atom ``x``. """
        return stream.Ty.sequence(stream.Ty.base(str(atom)))

    def box(self, box: markov.Box) -> stream.Stream:
        """ The sequence of boxes ``f0, f1, ...`` of a box ``f``. """
        return stream.Stream[markov.Diagram].sequence(
            box.name, self(box.dom), self(box.cod))

    def __call__(self, other):
        if isinstance(other, feedback.Wire) and other.time_step:
            state = str(other.reset())
            initial = stream.Ty.singleton(stream.Ty.base(state))
            return initial @ super().__call__(other)
        return super().__call__(other)


@dataclass
class Machine:
    """
    A cell :math:`I \\otimes S^d \\to O \\otimes S` from an input and the
    delayed state to an output and the next state, the uncurried coalgebra of
    a Mealy, Moore or unfolding recurrent network. Its feedback is the
    recurrent network, the unfold into the final coalgebra, see
    `Categorical Deep Learning <https://arxiv.org/abs/2402.15332>`_, Figure 1.
    Without parameters it is the unique map into it.

    The unrolled network takes the initial state :math:`S` once, as its first
    input, and the cell copies it where it needs to.

    The names of the boxes and of the atoms of the cell must not end with a
    digit: the atoms and boxes of the unrolling are named after their time
    step, and the functor to streams keeps no other record of it than that
    suffix, which :class:`Coalgebra` strips. A box reads the state only
    through its last wires, as many delayed copies of ``mem`` as it needs, and
    two boxes of the same name must have the same domain and codomain.

    Parameters:
        cell : The diagram from ``dom @ mem.delay()`` to ``cod @ mem``.
        dom : The input :math:`I`, empty for an unfold.
        cod : The output :math:`O`.
        mem : The state :math:`S`, not delayed.

    Raises:
        ValueError : If the cell is not of this shape.

    Example
    -------
    >>> from discopy.markov import Ty
    >>> I, O, S = Ty("I"), Ty("O"), Ty("S")
    >>> mealy = Machine.mealy(I, O, S)
    >>> assert mealy.cell.dom == I @ S.delay() and mealy.cell.cod == O @ S
    >>> unrolled = mealy.unroll(2)
    >>> print(unrolled.dom, "->", unrolled.cod)
    S @ I0 @ I1 @ I2 -> O0 @ O1 @ O2 @ S2
    >>> [box.name for box in unrolled.boxes if box.name.startswith("mealy")]
    ['mealy0', 'mealy1', 'mealy2']
    >>> moore = Machine.moore(I, O, S)
    >>> [box.name for box in moore.boxes]
    ['output', 'update']
    >>> assert moore.unroll(1).dom == Ty("S", "I0", "I1")
    >>> assert Machine.unfold(O, S).unroll(1).dom == S
    """
    cell: markov.Diagram
    dom: markov.Ty
    cod: markov.Ty
    mem: markov.Ty
    boxes: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        assert_isinstance(self.cell, markov.Diagram)
        if any(wire.time_step for wire in self.mem.inside):
            raise ValueError(f"The state {self.mem} must not be delayed.")
        shape = (self.dom @ self.mem.delay(), self.cod @ self.mem)
        if (self.cell.dom, self.cell.cod) != shape:
            raise ValueError(
                f"Expected a cell {shape[0]} -> {shape[1]}, got "
                f"{self.cell.dom} -> {self.cell.cod}.")
        self.boxes = self.generators()
        for box in self.boxes:
            self.reads(box)
        names = [box.name for box in self.boxes]
        names += [str(atom) for atom in self.atoms]
        for name in names:
            if name.endswith(tuple(digits)):
                raise ValueError(f"The name {name} ends with a digit.")

    def generators(self) -> tuple:
        """
        The distinct generating boxes of the cell, i.e. without its copies,
        discards and permutations, by name: the attribute :attr:`boxes`.

        Raises:
            ValueError : If two boxes of the same name differ in their domain
                or codomain.
        """
        boxes = {}
        for box in self.cell.boxes:
            if isinstance(box, markov.Function):
                continue
            first = boxes.setdefault(box.name, box)
            if (first.dom, first.cod) != (box.dom, box.cod):
                raise ValueError(
                    f"The boxes named {box.name} have different types: "
                    f"{first.dom} -> {first.cod} and {box.dom} -> {box.cod}.")
        return tuple(boxes.values())

    @property
    def atoms(self) -> tuple:
        """ The distinct atoms of the types of the cell, never delayed. """
        types = (self.dom, self.cod, self.mem) + tuple(
            ty for box in self.boxes for ty in (box.dom, box.cod))
        return tuple(dict.fromkeys(
            markov.Ty(wire.name) for ty in types for wire in ty.inside))

    def reads(self, box: markov.Box) -> int:
        """
        The number of times a box reads the delayed state, i.e. of copies of
        ``mem.delay()`` that end its domain.

        Parameters:
            box : A generating box of the cell.

        Raises:
            ValueError : If the box reads another delayed wire.
        """
        delayed = sum(1 for wire in box.dom.inside if wire.time_step)
        reads = delayed // len(self.mem) if self.mem else 0
        state = self.mem.delay() ** reads
        last = box.dom[len(box.dom) - len(state):]
        if delayed != len(state) or last != state:
            raise ValueError(
                f"The domain {box.dom} of {box.name} must end with "
                f"{reads} copies of {self.mem.delay()} and no other delayed "
                "wire.")
        return reads

    @classmethod
    def mealy(cls, dom: markov.Ty, cod: markov.Ty, mem: markov.Ty) -> Machine:
        """
        The Mealy machine :math:`S \\to (I \\to O \\times S)`: one box
        ``mealy`` from ``dom @ mem.delay()`` to ``cod @ mem``.

        Parameters:
            dom : The input :math:`I`.
            cod : The output :math:`O`.
            mem : The state :math:`S`.
        """
        cell = markov.Box("mealy", dom @ mem.delay(), cod @ mem)
        return cls(cell, dom, cod, mem)

    @classmethod
    def moore(cls, dom: markov.Ty, cod: markov.Ty, mem: markov.Ty) -> Machine:
        """
        The Moore machine :math:`S \\to O \\times (I \\to S)`: the box
        ``output`` from ``mem.delay()`` to ``cod`` and the box ``update`` from
        ``dom @ mem.delay()`` to ``mem`` read copies of the delayed state, so
        that the output at a time step does not depend on its input.

        Parameters:
            dom : The input :math:`I`.
            cod : The output :math:`O`.
            mem : The state :math:`S`.
        """
        state = mem.delay()
        output = markov.Box("output", state, cod)
        update = markov.Box("update", dom @ state, mem)
        cell = (dom @ markov.Diagram.copy(state) >> dom @ output @ state
                >> markov.Diagram.swap(dom, cod) @ state >> cod @ update)
        return cls(cell, dom, cod, mem)

    @classmethod
    def unfold(cls, cod: markov.Ty, mem: markov.Ty) -> Machine:
        """
        The unfolding machine :math:`S \\to O \\times S`: one box ``unfold``
        from ``mem.delay()`` to ``cod @ mem`` and no input.

        Parameters:
            cod : The output :math:`O`.
            mem : The state :math:`S`.
        """
        cell = markov.Box("unfold", mem.delay(), cod @ mem)
        return cls(cell, markov.Ty(), cod, mem)

    def feedback(self) -> markov.Diagram:
        """ The recurrent network: the delayed feedback of the cell. """
        return self.cell.feedback(dom=self.dom, cod=self.cod, mem=self.mem)

    def unroll(self, steps: int) -> markov.Diagram:
        """
        The first ``steps + 1`` time steps of the recurrent network, a finite
        diagram whose boxes carry their time step in their name. Its first
        input is the initial state, the state before the first step, which the
        boxes of the first step read through the copies of the cell.

        Parameters:
            steps : The number of steps after the first one.
        """
        functor = ToStream()
        initial = stream.Ty.singleton(
            stream.Ty.base(*(str(wire) for wire in self.mem.inside)))
        looped = functor(self.cell).feedback(
            dom=functor(self.dom) @ initial, cod=functor(self.cod),
            mem=functor(self.mem))
        unrolled = looped.unroll(steps).now
        size, state = len(self.dom), len(self.mem)
        if not size:
            return unrolled
        inputs = unrolled.dom
        swap = markov.Diagram.swap(inputs[size:size + state], inputs[:size])
        return swap @ inputs[size + state:] >> unrolled


@dataclass
class Coalgebra(Interpretation):
    """
    A coalgebra of a machine: one image for each box of its cell, sending
    every time step of a box to the same image, and the initial state
    ``start`` which a box reads at the first time step instead of the
    delayed state, a functor out of the unrolled machines.

    When ``cod`` is a category of parametric maps, the images and the start
    have the same parameter :math:`P`, which the unrolled image ties by one
    copy :math:`P \\to P^{\\otimes k}`, :math:`k` being the number of boxes of
    the unrolled diagram plus one for the start: the same cell with the same
    weights at every time step. The start is placed once, before the first
    step, and the copies of the cell, mapped to the copies of ``cod``, share
    it: for a stochastic start, one sample is read by every box that reads the
    state. The ``start`` is a field of its own, not an image of a box: a box
    may be named ``start``.

    The repr of a coalgebra is transparent only up to the maps it holds, as
    that of an :class:`~discopy.neural.datatype.Algebra`.

    Parameters:
        machine : The machine.
        ar_map : A map from the name of each box of the cell to its image,
            from the image of its domain to that of its codomain.
        ob_map : A map from the atoms of the cell to ``cod.ob``.
        cod : The codomain category, a Markov category.
        start : The arrow from the unit to the image of the state.

    Raises:
        ValueError : If a box or an atom has no image, if ``ar_map`` has a key
            that is not a box, if an image or the start has another boundary
            than expected or, in a parametric category, is not an arrow of
            ``cod``, if the parameters differ or if the parameter is empty.

    Example
    -------
    >>> from discopy import python
    >>> from discopy.markov import Ty
    >>> I, O, S = Ty("I"), Ty("O"), Ty("S")
    >>> Int = python.Ty(int)
    >>> F = Coalgebra(
    ...     Machine.mealy(I, O, S),
    ...     {"mealy": python.Function(
    ...         lambda a, s: (a + s, 2 * s + a), Int @ Int, Int @ Int)},
    ...     {I: Int, O: Int, S: Int}, python.Function,
    ...     python.Function(lambda: 1, python.Ty(), Int))
    >>> F(2)(1, 2, 3)
    (2, 5, 11, 19)
    """
    machine: Machine
    ar_map: dict
    ob_map: dict
    cod: type
    start: Any

    def __post_init__(self):
        self.check()

    def check(self):
        """
        Raise ``ValueError`` unless the boxes of the machine are exactly the
        keys of ``ar_map``, every atom of the machine has an image, in a
        parametric category the images and the start are arrows of ``cod``
        with the same non-empty parameter, and the boundaries of the images
        and of the start are the images of those of the boxes and of the
        state.
        """
        boxes = self.machine.boxes
        names = {box.name for box in boxes}
        for box in boxes:
            if box.name not in self.ar_map:
                raise ValueError(f"No image for the box {box.name}.")
        for name in self.ar_map:
            if name not in names:
                raise ValueError(f"No box {name} in the machine.")
        for atom in self.machine.atoms:
            if atom not in self.ob_map:
                raise ValueError(f"No image for the atom {atom}.")
        if self.parametric:
            images = {box: self.ar_map[box.name] for box in boxes}
            check_param(images | {"start": self.start}, self.cod)
            if not self.param:
                raise ValueError("Expected a non-empty parameter.")
        for box in boxes:
            self.check_boundary(box.name, self.ar_map[box.name], box.dom,
                                box.cod)
        self.check_boundary("start", self.start, markov.Ty(),
                            self.machine.mem)

    def ob(self, ty: markov.Ty):
        """
        The image of a type of the cell, delayed or not.

        Parameters:
            ty : A type of the machine.
        """
        return self.functor(markov.Ty(*(wire.name for wire in ty.inside)))

    def check_boundary(self, name: str, image, dom: markov.Ty, cod: markov.Ty):
        """
        Raise ``ValueError`` unless an image goes from the image of ``dom`` to
        the image of ``cod``.

        Parameters:
            name : The name of the box, for the error.
            image : The image of the box.
            dom : The domain of the box.
            cod : The codomain of the box.
        """
        if (image.dom, image.cod) != (self.ob(dom), self.ob(cod)):
            raise ValueError(
                f"The image of {name} goes from {image.dom} to {image.cod}, "
                f"expected {self.ob(dom)} -> {self.ob(cod)}.")

    @property
    def param(self):
        """ The common parameter of the images, ``None`` if not parametric. """
        return self.start.param if self.parametric else None

    def atom(self, atom: markov.Ty):
        """
        The image of an atom of an unrolled machine, i.e. of an atom of the
        cell at some time step.

        Parameters:
            atom : An atom of the cell followed by its time step.
        """
        return self.ob_map[markov.Ty(str(atom).rstrip(digits))]

    def image(self, box: markov.Box):
        """
        The image of a box of an unrolled machine: that of the box of the cell
        of the same name, whatever its time step.

        Parameters:
            box : A box ``name + str(time_step)``.
        """
        return self.ar_map[box.name.rstrip(digits)]

    @property
    def functor(self) -> markov.Functor:
        """ The functor from the unrolled machines. """
        return markov.Functor(
            ob_map=self.atom, ar_map=self.image, cod=self.cod)

    def __call__(self, steps: int):
        """
        The image of the unrolling of the machine to ``steps + 1`` time steps,
        with the ``start`` before the first step: one sample, which the cell
        copies. In a parametric category the copy of the parameter ties the
        images of the boxes of the unrolled diagram and the start.

        Parameters:
            steps : The number of steps after the first one.
        """
        unrolled = self.machine.unroll(steps)
        rest = self.cod.id(self.functor(unrolled.dom[len(self.machine.mem):]))
        image = self.start @ rest >> self.functor(unrolled)
        uses = sum(1 for box in unrolled.boxes
                   if not isinstance(box, markov.Function)) + 1
        return self.tie(image, uses)
