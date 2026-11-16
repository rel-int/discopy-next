"""
The two networks that read a picture, see :mod:`discopy.drawing.picture`:
one labels each pixel, the other says which module the picture is in.

Both pass messages on the grid of pixels, each pixel a node and its eight
neighbours its edges: a three by three convolution is one round of such
message passing, its weight a function of where the message comes from.
A box is told from the background only by the outline around it, so a
pixel needs a wide receptive field: rounds alternate with pooling, which
passes messages on a coarser grid, and the labels are read back up through
the finer ones, a U-Net (Ronneberger et al., MICCAI 2015).

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Rounds
    PixelNet
    ModuleNet

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        dataset
        train
        predict
        flood
        evaluate
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from discopy.drawing import picture
from discopy.drawing.picture import LABELS, MODULES


class Rounds(nn.Sequential):
    """ Rounds of message passing on the grid, each a three by three
    convolution, a group norm and a ReLU. """
    def __init__(self, inputs: int, outputs: int, rounds=2):
        super().__init__(*(layer for i in range(rounds) for layer in (
            nn.Conv2d(outputs if i else inputs, outputs, 3, padding=1),
            nn.GroupNorm(4, outputs), nn.ReLU())))


class PixelNet(nn.Module):
    """
    The label of each pixel, see :data:`picture.LABELS`: one round on the
    grid of pixels, then two on each grid twice as coarse, ``depth`` times,
    then back to the finest, each reading the finer grid it came from, with
    one round on the grid of pixels again.

    >>> net = PixelNet(width=8, depth=2)
    >>> net(torch.zeros(1, 3, 32, 32)).shape
    torch.Size([1, 23, 32, 32])
    """
    def __init__(self, width=8, depth=4, classes=len(LABELS)):
        super().__init__()
        widths = [width * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList(
            Rounds(a, b, 2 if i else 1)
            for i, (a, b) in enumerate(zip([3] + widths, widths)))
        self.up = nn.ModuleList(
            Rounds(2 * b + b, b, 2 if i < depth - 1 else 1)
            for i, b in enumerate(reversed(widths[:-1])))
        self.head = nn.Conv2d(width, classes, 1)

    def forward(self, x):
        skips = []
        for i, rounds in enumerate(self.down):
            x = rounds(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for rounds, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = rounds(torch.cat([x, skip], 1))
        return self.head(x)


class ModuleNet(nn.Module):
    """
    The module of a picture, one of :data:`picture.MODULES`: rounds on
    coarser and coarser grids, then the mean and the maximum of each
    channel over the whole picture.

    >>> ModuleNet(width=4, depth=2)(torch.zeros(2, 3, 32, 32)).shape
    torch.Size([2, 4])
    """
    def __init__(self, width=8, depth=5):
        super().__init__()
        widths = [width * 2 ** i for i in range(depth)]
        self.rounds = nn.ModuleList(
            Rounds(a, b, 2 if i else 1)
            for i, (a, b) in enumerate(zip([3] + widths, widths)))
        self.head = nn.Linear(2 * widths[-1], len(MODULES))

    def forward(self, x):
        for i, rounds in enumerate(self.rounds):
            x = rounds(F.max_pool2d(x, 2) if i else x)
        return self.head(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))


def generate(args):
    """ The samples of one module from one seed, for a worker. """
    module, seed, n, flags = args
    return [(module, ) + triple for triple in islice(
        picture.samples(module, seed, **flags), n)]


def dataset(n: int, seed=0, modules=MODULES, workers=4, **flags) -> dict:
    """
    ``n`` samples of each module, from seeds ``seed``, ``seed + 1``, ...
    one per hundred samples, drawn by ``workers`` processes: a dict of the
    ``images``, ``masks`` and ``modules`` as arrays, and their ``diagrams``.
    The flags are those of :func:`picture.samples`.
    """
    jobs = [(module, seed + i, min(100, n - 100 * i), flags)
            for module in modules for i in range((n + 99) // 100)]
    with ProcessPoolExecutor(workers) as pool:
        found = [x for chunk in pool.map(generate, jobs) for x in chunk]
    return {
        "images": np.stack([image for _, _, image, _ in found]),
        "masks": np.stack([mask for _, _, _, mask in found]),
        "modules": np.array([MODULES.index(m) for m, *_ in found]),
        "diagrams": [diagram for _, diagram, _, _ in found]}


def tensor(images: np.ndarray) -> torch.Tensor:
    """ Images as a float tensor of shape ``(n, 3, size, size)``. """
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255


def train(data: dict, seed=0, epochs=6, batch=16, lr=2e-3, log=print,
          pixel: PixelNet | None = None, module: ModuleNet | None = None,
          threads: int | None = None, checkpoint: str | None = None):
    """
    Train a :class:`PixelNet` and a :class:`ModuleNet` on a
    :func:`dataset`, with AdamW and a one-cycle schedule, background pixels
    weighted a tenth as much as the others, on ``threads`` threads: more
    than the free cores slows it down many times. Deterministic given
    ``seed`` and the threads. With a ``checkpoint`` path, the networks, the
    optimiser, the schedule and the random states are saved there after
    each epoch, and training resumes from it when it exists.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    pixel, module = pixel or PixelNet(), module or ModuleNet()
    parameters = list(pixel.parameters()) + list(module.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    steps = epochs * (len(data["images"]) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, lr, steps)
    weight = torch.ones(len(LABELS))
    weight[0] = .1
    rng, first = np.random.default_rng(seed), 0
    if checkpoint is not None and os.path.exists(checkpoint):
        state = torch.load(checkpoint, weights_only=False)
        for name, part in [("pixel", pixel), ("module", module),
                           ("optimiser", optimiser), ("schedule", schedule)]:
            part.load_state_dict(state[name])
        rng.bit_generator.state, first = state["rng"], state["epoch"] + 1
        torch.set_rng_state(state["torch"])
    for epoch in range(first, epochs):
        order, total = rng.permutation(len(data["images"])), 0.
        for i in range(0, len(order) - batch + 1, batch):
            index = order[i:i + batch]
            x = tensor(data["images"][index])
            masks = torch.from_numpy(data["masks"][index])
            modules = torch.from_numpy(data["modules"][index])
            loss = F.cross_entropy(pixel(x), masks, weight)\
                + F.cross_entropy(module(x), modules)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item()
        log(f"epoch {epoch}: loss {total / (len(order) // batch):.4f}")
        if checkpoint is not None:
            torch.save({
                "pixel": pixel.state_dict(), "module": module.state_dict(),
                "optimiser": optimiser.state_dict(),
                "schedule": schedule.state_dict(), "epoch": epoch,
                "rng": rng.bit_generator.state,
                "torch": torch.get_rng_state()}, checkpoint + ".part")
            os.replace(checkpoint + ".part", checkpoint)
    return pixel.eval(), module.eval()


@torch.no_grad()
def predict(pixel: PixelNet, module: ModuleNet, images: np.ndarray,
            batch=32) -> tuple[np.ndarray, np.ndarray]:
    """ The predicted masks and module indices of some images. """
    masks, modules = [], []
    for i in range(0, len(images), batch):
        x = tensor(images[i:i + batch])
        masks.append(pixel(x).argmax(1).numpy())
        modules.append(module(x).argmax(1).numpy())
    return np.concatenate(masks), np.concatenate(modules)


def flood(image: np.ndarray, threshold=128) -> np.ndarray:
    """
    The baseline mask of an image: a flood fill from its border through
    the light pixels is the background, the light pixels it does not reach
    are inside a box, named with the first of :data:`picture.NAMES`, and
    the dark pixels are wires, of the first of :data:`picture.TYPES`.
    """
    light = image.mean(axis=2) >= threshold
    reached = np.zeros_like(light)
    reached[[0, -1], :], reached[:, [0, -1]] = light[[0, -1], :], light[
        :, [0, -1]]
    while True:
        grown = reached.copy()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            grown |= picture.shift(reached, dr, dc, False)
        grown &= light
        if np.array_equal(grown, reached):
            break
        reached = grown
    mask = np.zeros(light.shape, dtype=np.int64)
    mask[~light] = LABELS.index(picture.TYPES[0])
    mask[light & ~reached] = LABELS.index(picture.NAMES[0])
    return mask


def recovered(mask, module: str, diagram) -> bool:
    """ Whether a mask decodes to a diagram, up to the module's axioms. """
    try:
        return picture.equal(picture.decode(mask, module), diagram, module)
    except (ValueError, IndexError, NotImplementedError, AssertionError):
        return False


def score(args) -> tuple[str, bool, bool]:
    """ The true module of a picture, whether its predicted mask decodes
    to its diagram, and whether its :func:`flood` does, for a worker. """
    diagram, image, mask, truth, guess = args
    return truth, recovered(mask, guess, diagram), recovered(
        flood(image), truth, diagram)


def evaluate(data: dict, masks: np.ndarray, modules: np.ndarray,
             workers=4) -> dict:
    """
    The exact recovery of the predicted masks decoded in the predicted
    modules, per true module, and the baseline's, a :func:`flood` decoded
    in the true module; the accuracy of the pixels per class and of the
    modules.
    """
    jobs = [(diagram, data["images"][i], masks[i],
             MODULES[data["modules"][i]], MODULES[modules[i]])
            for i, diagram in enumerate(data["diagrams"])]
    exact, baseline, count = Counter(), Counter(), Counter()
    with ProcessPoolExecutor(workers) as pool:
        for truth, ok, base in pool.map(score, jobs, chunksize=16):
            count[truth] += 1
            exact[truth] += ok
            baseline[truth] += base
    pixels = {LABELS[c]: float((masks[data["masks"] == c] == c).mean())
              for c in np.unique(data["masks"])}
    return {
        "exact": {m: exact[m] / count[m] for m in count},
        "baseline": {m: baseline[m] / count[m] for m in count},
        "pixels": pixels,
        "modules": float((modules == data["modules"]).mean())}
