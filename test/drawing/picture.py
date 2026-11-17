from itertools import islice

import numpy as np
from pytest import mark

from discopy import compact, frobenius
from discopy.drawing import picture
from discopy.drawing.picture import (
    LABELS, MODULES, augment, elementary, equal, kind, render, samples)


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
