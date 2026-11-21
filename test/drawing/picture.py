import shutil
from itertools import islice

import numpy as np
from pytest import mark, raises

from discopy import compact, frobenius
from discopy.drawing import picture
from discopy.drawing.picture import (
    COARSE, LABELS, MODULES, augment, coarsen, decode, elementary, equal,
    kind, read, render, samples, scan)


def test_elementary_draws_one_generator_at_a_time():
    x, y = map(frobenius.Ty, "ab")
    permutation = frobenius.Permutation(x @ x @ y, [2, 0, 1])
    diagram = permutation >> frobenius.Cospan(
        permutation.cod, [0, 1, 1], [1, 0])
    result = elementary(diagram)
    assert {kind(box) for box in result.boxes} <= {
        "swap", "cup", "cap", "spider"}
    assert all(len(layer.boxes_and_offsets) == 1 for layer in result.inside)
    assert equal(result, diagram, "frobenius")


def test_samples_are_deterministic():
    first, second = (list(islice(samples("compact", seed=3, size=64), 3))
                     for _ in range(2))
    for one, other in zip(first, second):
        assert one[0] == other[0]
        assert all(np.array_equal(a, b) for a, b in zip(one[1:], other[1:]))


@mark.parametrize("module", MODULES)
def test_samples_are_of_their_module(module):
    for diagram, image, mask in islice(samples(module, seed=1), 5):
        assert isinstance(diagram, picture.load(module).Diagram)
        assert len(diagram.boxes) <= 6
        assert image.shape == (160, 160, 3) and mask.shape == (160, 160)
        assert mask.max() < len(LABELS)


def test_mask_labels_what_is_drawn():
    x = compact.Ty('a')
    diagram = compact.Box('f', x, x @ x) >> compact.Swap(x, x)
    labels = {LABELS[i] for i in np.unique(render(diagram, 96, mask=True))}
    assert labels == {"background", "a", "f", "swap", "label a"}


def test_augment_moves_image_and_mask_together():
    x = compact.Ty('a')
    diagram = compact.Box('f', x, x)
    image, mask = render(diagram, 64), render(diagram, 64, mask=True)
    flags = dict.fromkeys(picture.AUGMENTATIONS, True)
    flags.pop("whiteboard")
    moved, labels = augment(image, mask, np.random.default_rng(0), **flags)
    assert moved.shape == image.shape and moved.dtype == np.uint8
    assert set(labels.flat) <= set(mask.flat)
    assert (labels == LABELS.index("f")).sum()\
        > .8 * (mask == LABELS.index("f")).sum()
    moved, labels = augment(image, mask, np.random.default_rng(0), warp=True,
                            rotate=True, perspective=True)
    ink = moved.mean(axis=2) < 128
    assert (labels[ink] != 0).mean() > .9


@mark.parametrize("flag", ["chalk", "whiteboard", "xkcd"])
def test_every_style_draws(flag):
    x = compact.Ty('a')
    diagram = compact.Box('f', x, x @ x) >> compact.Swap(x, x)
    plain = render(diagram, 64)
    image, _ = augment(plain, render(diagram, 64, mask=True),
                       np.random.default_rng(0), **{flag: True})\
        if flag != "xkcd" else (render(diagram, 64, xkcd=True), None)
    assert image.shape == (64, 64, 3)
    assert not np.array_equal(image, plain)


def read_true_mask(mask, module):
    def majority(rows, cols):
        return LABELS[np.bincount(mask[rows, cols]).argmax()]
    return read(coarsen(mask), majority,
                lambda rows, cols: majority(rows, cols).removeprefix("label "),
                module=module)


@mark.parametrize("module", MODULES)
def test_read_true_masks(module):
    """ A label drawn over a primitive, e.g. a spider on a short wire, is
    hidden in the mask, so a type can be lost: 990 to 1,000 of 1,000 read
    back per module from seed 7, the reading's ceiling (#559). This test
    reads the first 25 and allows one miss; the 1,000, some ten minutes,
    are the gate run on #559. """
    read_back = 0
    for diagram, _, mask in islice(samples(module, seed=7), 25):
        assert scan(coarsen(mask)) in MODULES[:MODULES.index(module) + 1]
        read_back += equal(read_true_mask(mask, module), diagram, module)
    assert read_back >= 24


def test_wire_labels_are_read_from_the_mask():
    x, y = compact.Ty('b'), compact.Ty('c')
    mask = render(compact.Box('f', x, y), 96, mask=True)
    labels = {LABELS[i] for i in np.unique(mask)}
    assert {"label b", "label c"} <= labels
    classes = coarsen(mask)
    assert set(np.unique(classes)) == {
        COARSE.index(c) for c in ("background", "wire", "box", "label")}


def test_decode_raises_on_what_is_not_a_picture():
    mask = np.zeros((32, 32), dtype=np.int64)
    mask[8:24, 8:24] = LABELS.index("swap")
    mask[:8, 15] = LABELS.index("a")
    with raises(ValueError):
        decode(mask, "symmetric")


def test_an_adjoint_type_reads_compact():
    x = compact.Ty('a')
    diagram = compact.Swap(x.r, x)
    mask = render(diagram, 96, mask=True)
    assert scan(coarsen(mask)) == "symmetric"
    assert equal(read_true_mask(mask, None), diagram, "compact")


def test_decode_joins_a_box_a_label_cuts():
    from discopy import symmetric
    x, y = symmetric.Ty('a'), symmetric.Ty('b')
    diagram = symmetric.Box('f', x, x @ y) >> symmetric.Swap(x, y)
    mask = render(diagram, 96, mask=True)
    rows, cols = np.nonzero(mask == LABELS.index("f"))
    middle = (cols.min() + cols.max()) // 2
    mask[rows.min():rows.max() + 1, middle:middle + 2] = LABELS.index(
        "label a")
    assert equal(decode(mask, "symmetric"), diagram, "symmetric")


def test_decode_types_both_legs_of_an_unlabelled_cup():
    x = compact.Ty('a')
    diagram = compact.Cup(x, x.r)
    mask = render(diagram, 96, mask=True, wire_labels=False)
    assert equal(read_true_mask(mask, None), diagram, "compact")


def test_read_a_self_dual_cup_in_frobenius():
    x = frobenius.Ty('a')
    diagram = frobenius.Cup(x, x)
    mask = render(diagram, 96, mask=True)
    assert scan(coarsen(mask)) == "compact"
    assert equal(read_true_mask(mask, None), diagram, "frobenius")


def test_decode_raises_on_a_primitive_outside_its_module():
    x = compact.Ty('a')
    mask = render(compact.Cup(x, x.r), 96, mask=True)
    with raises(ValueError):
        decode(mask, "symmetric")


def test_scan_ignores_scattered_pixels():
    classes = np.zeros((32, 32), dtype=np.uint8)
    classes[[2, 10, 20], [3, 15, 28]] = COARSE.index("swap")
    assert scan(classes) == "monoidal"


def test_decode_raises_on_a_self_dual_cup_in_monoidal():
    x = frobenius.Ty('a')
    mask = render(frobenius.Cup(x, x), 96, mask=True)
    with raises(ValueError):
        decode(mask, "monoidal")


def test_crop_owns_its_pixels():
    from discopy.drawing.picture import CROP, crop
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    result = crop(image, np.array([10, 20]), np.array([10, 30]))
    assert result.shape == CROP and result.dtype == np.float32
    assert result.base is None


def test_decode_least_moves_up_past_a_primitive():
    from discopy.drawing.picture import decode_least
    x = compact.Ty('a')
    diagram = compact.Cup(x, x.r)
    mask = render(diagram, 96, mask=True)
    assert equal(decode_least(mask, "monoidal"), diagram, "compact")


needs_graphviz = mark.skipif(
    shutil.which("dot") is None, reason="Graphviz is not installed.")


@needs_graphviz
def test_graphviz_keeps_the_port_order():
    x, y, z = map(frobenius.Ty, "abc")
    diagram = frobenius.Box('g', frobenius.Ty(), x)\
        @ frobenius.Box('h', frobenius.Ty(), y)\
        >> frobenius.Box('f', x @ y, z @ z) >> frobenius.Spider(2, 1, z)
    mask = render(diagram, 160, mask=True, graphviz=True)
    assert equal(decode(mask, "frobenius"), diagram, "frobenius")


@needs_graphviz
def test_graphviz_pictures_of_graphs_read_back():
    read_back = sum(equal(decode(mask, "symmetric"), diagram, "symmetric")
                    for diagram, _, mask in islice(samples(
                        "symmetric", seed=3, graphviz=True, graphs=True), 10))
    assert read_back >= 8


@needs_graphviz
def test_graphviz_lays_out_names_with_quotes_and_brackets():
    x = frobenius.Ty('a')
    names = {node.get("tooltip") for node in picture.layout(
        frobenius.Box('f"<g>&', x, x))["objects"]}
    assert 'f&quot;&lt;g&gt;&amp;' in names


@needs_graphviz
def test_graphviz_crosses_wires_at_swaps():
    cube = next(d for d in picture.graphs("symmetric", seed=0)
                if len(d.boxes) - sum(map(bool, map(kind, d.boxes))) == 8)
    graph = picture.layout(cube)
    assert picture.crossings(graph) == []
    assert any(node["name"][0] == "x" for node in graph["objects"])
    mask = render(cube, 320, mask=True, scale=12, graphviz=graph)
    assert equal(decode(mask, "symmetric"), cube, "symmetric")
