from itertools import islice

import numpy as np
from pytest import mark, raises

from discopy import compact, frobenius
from discopy.drawing import picture
from discopy.drawing.picture import (
    LABELS, MODULES, augment, decode, elementary, equal, kind, render,
    samples)


def test_elementary_draws_one_generator_at_a_time():
    x, y = map(frobenius.Ty, "ab")
    permutation = frobenius.Permutation(x @ x @ y, [2, 0, 1])
    diagram = permutation >> frobenius.Cospan(
        permutation.cod, [0, 1, 1], [1, 0])
    result = elementary(diagram)
    assert {kind(box) for box in result.boxes} <= {
        "swap", "cup", "cap", "spider"}
    assert equal(result, diagram, "frobenius")


def test_samples_are_deterministic():
    first, second = (next(samples("compact", seed=3, size=64))
                     for _ in range(2))
    assert first[0] == second[0]
    assert all(np.array_equal(a, b) for a, b in zip(first[1:], second[1:]))


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
    assert labels == {"background", "a", "f", "swap"}


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


@mark.parametrize("flag", ["chalk", "whiteboard", "xkcd"])
def test_every_style_draws(flag):
    _, image, _ = next(samples("symmetric", seed=2, size=64, **{flag: True}))
    assert image.shape == (64, 64, 3)


@mark.parametrize("module", MODULES)
def test_decode_true_masks(module):
    for diagram, _, mask in islice(samples(module, seed=11), 25):
        assert equal(decode(mask, module), diagram, module)


def test_decode_raises_on_what_is_not_a_picture():
    mask = np.zeros((32, 32), dtype=np.int64)
    mask[8:24, 8:24] = LABELS.index("swap")
    mask[:8, 15] = LABELS.index("a")
    with raises(ValueError):
        decode(mask, "symmetric")
