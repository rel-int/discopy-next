from random import Random

import pytest

import discopy
from discopy import markov, para, python
from discopy.markov import Box, Diagram, Ty
from discopy.neural.machine import Coalgebra, Machine
from fixtures.datatype import Int, Net, P, Para, S, check_grads

I, O, T = Ty("I"), Ty("O"), Ty("T")
MEALY, MOORE = Machine.mealy(I, O, S), Machine.moore(I, O, S)
UNFOLD = Machine.unfold(O, S)
TWO = Machine(Box("f", I @ S.delay() @ T.delay(), O @ S @ T), I, O, S @ T)
IGNORES = Machine(Box("gen", I @ S.delay(), O @ S), I, O, S)
READS = Machine(
    I @ Diagram.copy(S.delay(), 3)
    >> Box("triple", I @ S.delay() @ S.delay() @ S.delay(), O @ S), I, O, S)


def steps(name, n):
    """ The types ``name0 @ ... @ name_n`` of an unrolled machine. """
    return Ty(*(f"{name}{t}" for t in range(n + 1)))


def named(diagram, name):
    return {box.name: box for box in diagram.boxes
            if box.name.startswith(name)}


# 1. machines: Mealy, Moore and unfolds


@pytest.mark.parametrize("n", range(4))
def test_mealy_unroll(n):
    unrolled = MEALY.unroll(n)
    boxes = named(unrolled, "mealy")
    assert list(boxes) == [f"mealy{t}" for t in range(n + 1)]
    assert (unrolled.dom, unrolled.cod) == (
        S @ steps("I", n), steps("O", n) @ Ty(f"S{n}"))
    assert boxes["mealy0"].dom == Ty("I0", "S")  # the initial state is read
    assert boxes["mealy0"].cod == Ty("O0", "S0")
    for t in range(1, n + 1):
        box = boxes[f"mealy{t}"]
        assert box.dom == Ty(f"I{t}", f"S{t - 1}")
        assert box.cod == Ty(f"O{t}", f"S{t}")


@pytest.mark.parametrize("n", range(4))
def test_moore_unroll(n):
    unrolled = MOORE.unroll(n)
    outputs, updates = named(unrolled, "output"), named(unrolled, "update")
    assert list(outputs) == [f"output{t}" for t in range(n + 1)]
    assert list(updates) == [f"update{t}" for t in range(n + 1)]
    assert (unrolled.dom, unrolled.cod) == (
        S @ steps("I", n), steps("O", n) @ Ty(f"S{n}"))
    assert outputs["output0"].dom == S  # one initial state, copied by the cell
    assert updates["update0"].dom == Ty("I0", "S")
    for t in range(1, n + 1):
        assert outputs[f"output{t}"].dom == Ty(f"S{t - 1}")
        assert outputs[f"output{t}"].cod == Ty(f"O{t}")
        assert updates[f"update{t}"].dom == Ty(f"I{t}", f"S{t - 1}")
        assert updates[f"update{t}"].cod == Ty(f"S{t}")


@pytest.mark.parametrize("n", range(4))
def test_unfold_unroll(n):
    unrolled = UNFOLD.unroll(n)
    boxes = named(unrolled, "unfold")
    assert list(boxes) == [f"unfold{t}" for t in range(n + 1)]
    assert (unrolled.dom, unrolled.cod) == (
        S, steps("O", n) @ Ty(f"S{n}"))
    assert boxes["unfold0"].dom == S
    for t in range(1, n + 1):
        assert boxes[f"unfold{t}"].dom == Ty(f"S{t - 1}")
        assert boxes[f"unfold{t}"].cod == Ty(f"O{t}", f"S{t}")


def test_machine_cells():
    assert MEALY.cell == Box("mealy", I @ S.delay(), O @ S)
    assert UNFOLD.cell == Box("unfold", S.delay(), O @ S)
    output = Box("output", S.delay(), O)
    update = Box("update", I @ S.delay(), S)
    assert MOORE.cell == (
        I @ Diagram.copy(S.delay()) >> I @ output @ S.delay()
        >> Diagram.swap(I, O) @ S.delay() >> O @ update)
    for machine in (MEALY, MOORE, UNFOLD):
        assert machine.feedback() == machine.cell.feedback(
            dom=machine.dom, cod=machine.cod, mem=machine.mem)
    assert [box.name for box in MOORE.boxes] == ["output", "update"]
    named = {box.name: box for box in MOORE.boxes}
    assert MEALY.boxes == (MEALY.cell, )
    assert (MOORE.reads(named["output"]), MOORE.reads(named["update"])) == (
        1, 1)
    assert set(MEALY.atoms) == {I, O, S} == set(MOORE.atoms)


def test_machine_repr():
    scope = vars(markov) | vars(discopy) | {"Machine": Machine}
    for machine in (MEALY, MOORE, UNFOLD):
        assert eval(repr(machine), scope) == machine


def test_machine_errors():
    with pytest.raises(ValueError, match="Expected a cell"):
        Machine(Box("f", I @ S, O @ S), I, O, S)  # the state is not delayed
    with pytest.raises(ValueError, match="Expected a cell"):
        Machine(Box("f", I @ S.delay(), O @ S), I, O, I)
    with pytest.raises(ValueError, match="digit"):  # time steps are digits
        Machine.mealy(I, O, Ty("S1"))
    with pytest.raises(ValueError, match="digit"):
        Machine.mealy(I, O, Ty("h2"))
    with pytest.raises(ValueError, match="digit"):
        Machine(Box("f1", I @ S.delay(), O @ S), I, O, S)
    with pytest.raises(ValueError, match="digit"):
        Machine(Box("layer10", I @ S.delay(), O @ S), I, O, S)
    with pytest.raises(ValueError, match="must not be delayed"):
        Machine(Box("f", I @ S.delay().delay(), O @ S.delay()),
                I, O, S.delay())
    same_name = Box("g", I @ S.delay(), I @ S.delay()) >> Box(
        "g", I @ S.delay(), O @ S)
    with pytest.raises(ValueError, match="different types"):
        Machine(same_name, I, O, S)
    state_first = Box("f", S.delay() @ I, O @ S)
    with pytest.raises(ValueError, match="delayed"):  # state before the input
        Machine(Diagram.swap(I, S.delay()) >> state_first, I, O, S)
    other = Box("g", I @ S.delay(), I @ I.delay() @ S.delay())
    with pytest.raises(ValueError, match="delayed"):  # another delayed wire
        Machine(other >> Box("f", I @ I.delay() @ S.delay(), O @ S), I, O, S)
    with pytest.raises(TypeError):
        Machine("cell", I, O, S)


def test_hand_written_cells_unroll():
    assert TWO.reads(TWO.boxes[0]) == 1 and TWO.mem == S @ T
    assert TWO.unroll(1).dom == Ty("S", "T", "I0", "I1")
    assert IGNORES.unroll(1).dom == Ty("S", "I0", "I1")
    assert READS.reads(READS.boxes[0]) == 3
    assert READS.unroll(1).dom == Ty("S", "I0", "I1")


# 2. coalgebras into parametric diagrams

OBS = {I: I, O: O, S: S}
START = Para(Ty(), S, Box("start", P, S), param=P)
STEP = Para(I @ S, O @ S, Box("step", I @ S @ P, O @ S), param=P)
OUTPUT = Para(S, O, Box("out", S @ P, O), param=P)
UPDATE = Para(I @ S, S, Box("upd", I @ S @ P, S), param=P)
UNFOLDING = Para(S, O @ S, Box("unf", S @ P, O @ S), param=P)
TRIPLE = Para(
    I @ S ** 3, O @ S, Box("triple", I @ S ** 3 @ P, O @ S), param=P)
DIAGRAMS = {
    "mealy": (MEALY, {"mealy": STEP}, {"step": 1}),
    "moore": (MOORE, {"output": OUTPUT, "update": UPDATE},
              {"out": 1, "upd": 1}),
    "unfold": (UNFOLD, {"unfold": UNFOLDING}, {"unf": 1}),
    "reads": (READS, {"triple": TRIPLE}, {"triple": 1})}


def count(diagram, name):
    return len([box for box in diagram.inside.boxes if box.name == name])


@pytest.mark.parametrize("n", range(5))
@pytest.mark.parametrize("kind", DIAGRAMS)
def test_diagram_coalgebra(kind, n):
    machine, images, boxes = DIAGRAMS[kind]
    coalgebra = Coalgebra(machine, images, OBS, Para, START)
    unrolled = machine.unroll(n)
    uses = (n + 1) * sum(boxes.values()) + 1  # the cell's boxes, one start
    untied = (START @ Para.id(coalgebra.functor(unrolled.dom[1:]))
              >> coalgebra.functor(unrolled))
    tied = coalgebra(n)
    assert untied.param == P ** uses
    assert tied.param == P and (tied.dom, tied.cod) == (
        machine.dom ** (n + 1), machine.cod ** (n + 1) @ S)
    assert tied == untied.reparam(Diagram.copy(P, uses))
    for name, per_step in boxes.items():
        assert count(tied, name) == per_step * (n + 1)
    assert count(tied, "start") == 1  # sampled once, copied by the cell


def test_coalgebra_start_before_step_zero():
    """ The start, then the step, once per time step. """
    coalgebra = Coalgebra(MEALY, {"mealy": STEP}, OBS, Para, START)
    two = coalgebra(1)
    assert two.dom == I @ I
    assert [box.name for box in two.inside.boxes if box.name in (
        "start", "step")] == ["start", "step", "step"]


def test_box_named_start():
    """ The start is a field, not an entry of ``ar_map``. """
    machine = Machine(Box("start", I @ S.delay(), O @ S), I, O, S)
    coalgebra = Coalgebra(machine, {"start": STEP}, OBS, Para, START)
    tied = coalgebra(2)
    assert count(tied, "start") == 1 and count(tied, "step") == 3
    assert tied.param == P and tied == Coalgebra(
        MEALY, {"mealy": STEP}, OBS, Para, START)(2)


def test_coalgebra_errors():
    with pytest.raises(ValueError, match="No image for the box"):
        Coalgebra(MOORE, {"output": OUTPUT}, OBS, Para, START)
    with pytest.raises(ValueError, match="No box other"):
        Coalgebra(MEALY, {"mealy": STEP, "other": STEP}, OBS, Para, START)
    with pytest.raises(ValueError, match="No image for the atom"):
        Coalgebra(MEALY, {"mealy": STEP}, {I: I, S: S}, Para, START)
    two = Para(Ty(), S, Box("two", P @ P, S), param=P @ P)
    with pytest.raises(ValueError, match="Expected one parameter"):
        Coalgebra(MEALY, {"mealy": STEP}, OBS, Para, two)
    with pytest.raises(ValueError, match="not an arrow"):
        Coalgebra(MEALY, {"mealy": STEP}, OBS, Para, Box("s", Ty(), S))


def test_coalgebra_boundaries():
    wrong_step = Para(I, O @ S, Box("step", I @ P, O @ S), param=P)
    with pytest.raises(ValueError, match="image of mealy goes from"):
        Coalgebra(MEALY, {"mealy": wrong_step}, OBS, Para, START)
    wrong_start = Para(Ty(), O, Box("start", P, O), param=P)
    with pytest.raises(ValueError, match="image of start goes from"):
        Coalgebra(MEALY, {"mealy": STEP}, OBS, Para, wrong_start)
    with pytest.raises(ValueError, match="image of mealy goes from"):
        Coalgebra(MEALY, {"mealy": STEP}, {I: I, O: O, S: O}, Para, START)


def test_coalgebra_empty_parameter():
    step = Para(I @ S, O @ S, Box("step", I @ S, O @ S), param=Ty())
    start = Para(Ty(), S, Box("zero", Ty(), S), param=Ty())
    with pytest.raises(ValueError, match="non-empty parameter"):
        Coalgebra(MEALY, {"mealy": step}, OBS, Para, start)


# 3. coalgebras into python functions, against plain loops


def python_images(kind):
    """ Integer cells, as parametric functions and as plain loops. """
    function = python.Function
    start = Net(python.Ty(), Int, function(lambda p: p + 1, Int, Int),
                param=Int)
    if kind == "mealy":
        images = {"mealy": Net(Int @ Int, Int @ Int, function(
            lambda a, s, p: (a * p + s, s * p - a), Int ** 3, Int @ Int),
            param=Int)}
    if kind == "moore":
        images = {
            "output": Net(Int, Int, function(
                lambda s, p: s * p - 1, Int @ Int, Int), param=Int),
            "update": Net(Int @ Int, Int, function(
                lambda a, s, p: a + s * p, Int ** 3, Int), param=Int)}
    if kind == "unfold":
        images = {"unfold": Net(Int, Int @ Int, function(
            lambda s, p: (s + p, 2 * s - p), Int @ Int, Int @ Int),
            param=Int)}
    return Coalgebra(
        {"mealy": MEALY, "moore": MOORE, "unfold": UNFOLD}[kind], images,
        {I: Int, O: Int, S: Int}, Net, start)


def loop(kind, xs, p):
    """ The recurrent network as a Python loop: the state starts at p + 1. """
    state, outs = p + 1, []
    for a in xs:
        if kind == "mealy":
            out, state = a * p + state, state * p - a
        elif kind == "moore":  # the output reads the state before the input
            out, state = state * p - 1, a + state * p
        else:
            out, state = state + p, 2 * state - p
        outs.append(out)
    return (*outs, state)


@pytest.mark.parametrize("kind", ["mealy", "moore", "unfold"])
def test_python_machines(kind):
    rng, coalgebra = Random(7), python_images(kind)
    for n in range(6):
        xs, p = [rng.randint(-5, 5) for _ in range(n + 1)], rng.randint(-4, 4)
        given = xs if coalgebra.machine.dom else []
        assert coalgebra(n).inside(*given, p) == loop(kind, xs, p)


def test_moore_output_reads_the_state_before_the_input():
    rng, coalgebra = Random(9), python_images("moore")
    for _ in range(20):
        xs, p = [rng.randint(-5, 5) for _ in range(5)], rng.randint(-4, 4)
        outs = coalgebra(4).inside(*xs, p)[:-1]
        for t in range(5):  # changing the inputs from t on keeps outputs 0..t
            ys = xs[:t] + [x + 1 for x in xs[t:]]
            assert coalgebra(4).inside(*ys, p)[:t + 1] == outs[:t + 1]


def test_python_machine_without_parameter():
    function = python.Function
    mealy = Coalgebra(
        MEALY, {"mealy": function(
            lambda a, s: (a + s, 2 * s + a), Int @ Int, Int @ Int)},
        {I: Int, O: Int, S: Int}, function,
        function(lambda: 1, python.Ty(), Int))
    assert mealy.param is None and not mealy.parametric
    state, outs = 1, []
    for a in (1, 2, 3):
        outs.append(a + state)
        state = 2 * state + a
    assert mealy(2)(1, 2, 3) == (*outs, state)


def run_two(xs):
    (s, t), outs = (1, 2), []
    for a in xs:
        outs.append(a + s)
        s, t = 2 * s + t, t - a
    return (*outs, s, t)


def run_ignores(xs):
    return (*[a + 1 for a in xs], 3 * xs[-1])


def run_reads(xs):
    state, outs = 2, []
    for a in xs:
        outs.append(a + 6 * state)
        state = state ** 2 - state
    return (*outs, state)


CELLS = [
    (TWO, lambda a, s, t: (a + s, 2 * s + t, t - a),
     python.Function(lambda: (1, 2), python.Ty(), Int @ Int), run_two),
    (IGNORES, lambda a, s: (a + 1, 3 * a),
     python.Function(lambda: 7, python.Ty(), Int), run_ignores),
    (READS, lambda a, r, s, t: (a + r + 2 * s + 3 * t, r * s - t),
     python.Function(lambda: 2, python.Ty(), Int), run_reads)]


@pytest.mark.parametrize("n", range(4))
@pytest.mark.parametrize("machine, image, start, reference", CELLS)
def test_python_hand_written_cells(machine, image, start, reference, n):
    """ Cells on the integers, without parameter, against plain loops. """
    function, rng = python.Function, Random(10 + n)
    coalgebra = Coalgebra(
        machine, {box.name: function(image, Int ** len(box.dom),
                                     Int ** len(box.cod))
                  for box in machine.boxes},
        {I: Int, O: Int, S: Int, T: Int}, function, start)
    xs = [rng.randint(-5, 5) for _ in range(n + 1)]
    assert coalgebra(n)(*xs) == reference(xs)


# 4. coalgebras into networks, run by torch


def test_network_mealy():
    torch = pytest.importorskip("torch")
    from discopy.neural.network import Box as Layer, Dim, Dims, Network
    from discopy.neural.torch import Module
    torch.manual_seed(2)
    x, h, y = Dims(3), Dims(5), Dims(2)
    H0, W = Dims(5), Dims(Dim(8, 5))

    def cell(a, s, w):
        new = torch.tanh(torch.cat([a, s], -1) @ w)
        return new[..., :2], new

    Nets = para.Markov[Network]
    step = Nets(x @ h, y @ h, param=H0 @ W, inside=(
        x @ h @ Network.discard(H0) @ W
        >> Layer("step", x @ h @ W, y @ h, module=cell)))
    start = Nets(Dims(), h, param=H0 @ W, inside=(
        Network.id(H0) @ Network.discard(W)))
    coalgebra = Coalgebra(
        MEALY, {"mealy": step}, {I: x, O: y, S: h}, Nets, start)
    for n in range(5):
        tied = coalgebra(n)
        module = Module(tied, torch.randn(5), torch.randn(8, 5) / 3)
        assert [tuple(p.shape) for p in module.parameters()] == [(5, ), (8, 5)]
        xs = [torch.randn(3) for _ in range(n + 1)]
        outs = module(*xs)
        s0, w = [p.detach().clone().requires_grad_() for p in module.weights]
        s, expected = s0, []
        for a in xs:
            out, s = cell(a, s, w)
            expected.append(out)
        expected.append(s)
        assert len(outs) == n + 2
        assert all(torch.allclose(o, e, atol=1e-6)
                   for o, e in zip(outs, expected))
        sum(o.sum() for o in outs).backward()
        sum(e.sum() for e in expected).backward()
        check_grads(module, (s0, w))


# 5. consistency with the trace of a network


@pytest.mark.parametrize("rounds", range(1, 5))
def test_unrolled_mealy_is_the_trace(rounds):
    torch = pytest.importorskip("torch")
    from discopy.neural.network import Box as Layer, Dims, Network
    torch.manual_seed(3)
    x, h, y = Dims(3), Dims(5), Dims(2)

    def cell(a, s):  # the output is not the state: its first two entries
        new = torch.tanh(a.sum(-1, keepdim=True) + s)
        return new[..., :2], new

    layer = Layer("cell", x @ h, y @ h, module=cell)
    zero = Layer("zero", Dims(), h, module=lambda: torch.zeros(5))
    coalgebra = Coalgebra(
        MEALY, {"mealy": layer}, {I: x, O: y, S: h}, Network, zero)
    a = torch.randn(3)
    traced = layer.trace().to_function(torch.Tensor, rounds=rounds)(a)
    outs = coalgebra(rounds - 1).to_function(torch.Tensor)(*[a] * rounds)
    state, expected = torch.zeros(5), []
    for _ in range(rounds):
        out, state = cell(a, state)
        expected.append(out)
    assert len(outs) == rounds + 1
    assert torch.allclose(traced, out, atol=1e-6)
    assert torch.allclose(outs[rounds - 1], traced, atol=1e-6)
    assert all(torch.allclose(o, e, atol=1e-6)
               for o, e in zip(outs, expected))
    assert torch.allclose(outs[rounds], state, atol=1e-6)  # the final state
