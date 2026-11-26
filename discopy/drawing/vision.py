"""
The networks that read a picture, see :mod:`~discopy.drawing.picture`: one
says what each pixel is, the other reads the name of each box and the type
of each wire from the text drawn on it.

The pixel net passes messages on the grid of pixels, each pixel a node and
its eight neighbours its edges: a three by three convolution is one round
of such message passing, its weight a function of where the message comes
from. A box is told from the background only by the outline around it, so
a pixel needs a wide receptive field: rounds alternate with pooling, which
passes messages on a coarser grid, and the classes are read back up through
the finer ones, a U-Net (Ronneberger et al., MICCAI 2015). It predicts the
classes of :data:`picture.COARSE`, no name and no type.

The names and types are read by a small OCR model, one for both, on a crop
of each box and of each wire label the pixel net found: convolutions, a
recurrent layer along the text and connectionist temporal classification
over its characters (Graves et al., ICML 2006), so a type such as ``b.r``
is read as three characters and needs no class of its own. No network says
which module a picture is in: :func:`picture.scan` reads it off the
structural primitives the pixel net found.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Rounds
    PixelNet
    OCR

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        dataset
        texts
        train
        train_ocr
        predict
        read
        save
        load
        from_picture
        flood
        evaluate
"""

from __future__ import annotations

import os
from collections import Counter
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from discopy.drawing import picture
from discopy.drawing.picture import (
    CHARS, COARSE, CROP, LABELS, MODULES, NAMES, TYPES, crop, decode_least,
    refine, scan)
from discopy.utils import AxiomError


class Rounds(nn.Sequential):
    """ Rounds of message passing on the grid, each a three by three
    convolution, a group norm and a ReLU. """
    def __init__(self, inputs: int, outputs: int, rounds=2):
        super().__init__(*(layer for i in range(rounds) for layer in (
            nn.Conv2d(outputs if i else inputs, outputs, 3, padding=1),
            nn.GroupNorm(4, outputs), nn.ReLU())))


class PixelNet(nn.Module):
    """
    The class of each pixel, see :data:`picture.COARSE`: one round on the
    grid of pixels, then two on each grid twice as coarse, ``depth`` times,
    then back to the finest, each reading the finer grid it came from, with
    one round on the grid of pixels again.

    >>> net = PixelNet(width=8, depth=2)
    >>> net(torch.zeros(1, 3, 32, 32)).shape
    torch.Size([1, 8, 32, 32])
    """
    def __init__(self, width=8, depth=4, classes=len(COARSE)):
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


class OCR(nn.Module):
    """
    The text on a crop of shape :data:`CROP`: convolutions, each with a
    batch norm, down to a quarter of its height, a bidirectional GRU along
    its width, and for each of its
    columns a score per character of :data:`CHARS` and the blank.

    >>> ocr = OCR(width=8)
    >>> ocr(torch.zeros(2, 1, *CROP)).shape
    torch.Size([2, 24, 16])
    >>> len(ocr.read(torch.zeros(2, 1, *CROP)))
    2
    """
    def __init__(self, width=32):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width), nn.ReLU())
        features = 2 * width * CROP[0] // 4
        self.rnn = nn.GRU(features, width, batch_first=True,
                          bidirectional=True)
        self.head = nn.Linear(2 * width, len(CHARS) + 1)

    def forward(self, x):
        x = self.convs(x).permute(0, 3, 1, 2).flatten(2)
        return self.head(self.rnn(x)[0])

    @torch.no_grad()
    def read(self, x) -> list[str]:
        """ The text on each crop, its best path with repeats and blanks
        removed. """
        best = self(x).argmax(-1).tolist()
        return ["".join(CHARS[c - 1] for k, c in enumerate(path)
                        if c and (k == 0 or c != path[k - 1]))
                for path in best]


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
        "masks": np.stack([mask for _, _, _, mask in found]).astype(np.uint8),
        "modules": np.array([MODULES.index(m) for m, *_ in found]),
        "diagrams": [diagram for _, diagram, _, _ in found]}


def tensor(images: np.ndarray) -> torch.Tensor:
    """ Images as a float tensor of shape ``(n, 3, size, size)``. """
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255


def truths(mask: np.ndarray):
    """ The components of the true classes of a mask, see
    :func:`picture.regions`, the name of each box and the type each label
    reads, by majority. """
    classes = picture.coarsen(mask)
    boxes, wires, labels = picture.regions(classes)

    def majority(ids, i):
        return LABELS[np.bincount(mask[ids == i]).argmax()]
    names = {int(b): majority(boxes, b) for b in np.unique(boxes[boxes >= 0])}
    reads = {int(t): majority(labels, t).removeprefix("label ")
             for t in np.unique(labels[labels >= 0])}
    return classes, boxes, wires, labels, names, reads


def cut(image: np.ndarray, ids: np.ndarray, i: int, label=False):
    """ The :func:`picture.crop` of the component ``i`` of ``ids``. """
    rows, cols = np.nonzero(ids == i)
    return crop(image, rows, cols, label)


def texts(data: dict, label: bool) -> tuple[np.ndarray, list[str]]:
    """ The crops of every box, or of every wire label, of a
    :func:`dataset` and the text each shows, from its masks. """
    crops, strings = [], []
    for image, mask in zip(data["images"], data["masks"]):
        _, boxes, _, labels, names, reads = truths(mask)
        ids, found = (labels, reads) if label else (boxes, names)
        for i, text in found.items():
            if text in (TYPES if label else NAMES):
                crops.append(cut(image, ids, i, label))
                strings.append(text)
    return np.stack(crops).astype(np.float32) if crops else np.zeros(
        (0, *CROP), np.float32), strings


def train(data: dict, seed=0, epochs=6, batch=16, lr=2e-3, log=print,
          pixel: PixelNet | None = None, threads: int | None = None,
          checkpoint: str | None = None) -> PixelNet:
    """
    Train a :class:`PixelNet` on the classes of the masks of a
    :func:`dataset`, with AdamW and a one-cycle schedule, background pixels
    weighted a tenth as much as the others, on ``threads`` threads: more
    than the free cores slows it down many times. Deterministic given
    ``seed`` and the threads. With a ``checkpoint`` path, the network, the
    optimiser, the schedule and the random states are saved there after
    each epoch, and training resumes from it when it exists.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    pixel = pixel or PixelNet()
    optimiser = torch.optim.AdamW(
        pixel.parameters(), lr=lr, weight_decay=1e-4)
    if not 1 <= batch <= len(data["images"]):
        raise ValueError(
            f"Expected a batch from 1 to {len(data['images'])}, got {batch}.")
    steps = epochs * (len(data["images"]) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, lr, steps)
    weight = torch.ones(len(COARSE))
    weight[0] = .1
    rng, first = np.random.default_rng(seed), 0
    if checkpoint is not None and os.path.exists(checkpoint):
        state = torch.load(checkpoint, weights_only=False)
        for name, part in [("pixel", pixel), ("optimiser", optimiser),
                           ("schedule", schedule)]:
            part.load_state_dict(state[name])
        rng.bit_generator.state, first = state["rng"], state["epoch"] + 1
        torch.set_rng_state(state["torch"])
    for epoch in range(first, epochs):
        order, total = rng.permutation(len(data["images"])), 0.
        for i in range(0, len(order) - batch + 1, batch):
            index = order[i:i + batch]
            classes = picture.coarsen(data["masks"][index])
            loss = F.cross_entropy(pixel(tensor(data["images"][index])),
                                   torch.from_numpy(classes), weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item()
        log(f"epoch {epoch}: loss {total / (len(order) // batch):.4f}")
        if checkpoint is not None:
            torch.save({
                "pixel": pixel.state_dict(),
                "optimiser": optimiser.state_dict(),
                "schedule": schedule.state_dict(), "epoch": epoch,
                "rng": rng.bit_generator.state,
                "torch": torch.get_rng_state()}, checkpoint + ".part")
            os.replace(checkpoint + ".part", checkpoint)
    return pixel.eval()


def train_ocr(crops: np.ndarray, strings: list[str], seed=0, epochs=4,
              batch=128, lr=2e-3, log=print, ocr: OCR | None = None) -> OCR:
    """ Train an :class:`OCR` on crops and their texts, with Adam and the
    connectionist temporal classification loss. """
    torch.manual_seed(seed)
    ocr = (ocr or OCR()).train()
    optimiser = torch.optim.Adam(ocr.parameters(), lr=lr)
    targets = [torch.tensor([CHARS.index(c) + 1 for c in s]) for s in strings]
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order, total = rng.permutation(len(crops)), 0.
        for i in range(0, len(order), batch):
            index = order[i:i + batch]
            scores = ocr(torch.from_numpy(crops[index]).unsqueeze(1))
            log_probs = scores.log_softmax(-1).transpose(0, 1)
            target = [targets[j] for j in index]
            loss = F.ctc_loss(
                log_probs, torch.cat(target),
                torch.full((len(index), ), log_probs.shape[0]),
                torch.tensor([len(t) for t in target]), zero_infinity=True)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(index)
        log(f"ocr epoch {epoch}: loss {total / len(order):.4f}")
    return ocr.eval()


@torch.no_grad()
def predict(pixel: PixelNet, images: np.ndarray, batch=32) -> np.ndarray:
    """ The predicted classes of the pixels of some images. """
    return np.concatenate([
        pixel(tensor(images[i:i + batch])).argmax(1).numpy()
        for i in range(0, len(images), batch)]).astype(np.uint8)


def read(ocr: OCR, image: np.ndarray, classes: np.ndarray):
    """ The components of the classes of an image's pixels, see
    :func:`picture.regions`, the name the OCR model reads in each box and
    the type it reads on each labelled wire. """
    boxes, wires, labels = picture.regions(classes)
    box_ids = [int(b) for b in np.unique(boxes[boxes >= 0])]
    attached = picture.attach(labels, wires)
    crops = [cut(image, boxes, b) for b in box_ids] + [
        cut(image, labels, t, True) for t in attached]
    found = ocr.read(torch.from_numpy(np.stack(crops).astype(
        np.float32)).unsqueeze(1)) if crops else []
    names = dict(zip(box_ids, found))
    types = {wire: found[len(box_ids) + k]
             for k, wire in enumerate(attached.values())}
    return boxes, wires, names, types


def save(path: str, pixel: PixelNet, ocr: OCR):
    """ Save the weights of the two networks to one file, with the widths
    and depth they were built with. """
    torch.save({"pixel": pixel.state_dict(), "ocr": ocr.state_dict(),
                "pixel_width": pixel.head.in_channels,
                "pixel_depth": len(pixel.down) - 1,
                "ocr_width": ocr.rnn.hidden_size}, path)


def load(path: str) -> tuple[PixelNet, OCR]:
    """ The two networks whose weights :func:`save` wrote to a file. """
    state = torch.load(path, weights_only=False)
    pixel = PixelNet(state["pixel_width"], state["pixel_depth"])
    ocr = OCR(state["ocr_width"])
    pixel.load_state_dict(state["pixel"])
    ocr.load_state_dict(state["ocr"])
    return pixel.eval(), ocr.eval()


def from_picture(jpeg, config: dict):
    """
    The diagram a picture shows, see :func:`~discopy.drawing.from_picture`.

    Parameters:
        jpeg : The picture, a path or its bytes, in any format Pillow reads.
        config : ``weights``, a path :func:`save` wrote or a pair of
            networks; ``scale``, the pixels per unit of the picture, if it
            is not the 24 the networks were trained at; ``size``, the side
            of the square the picture is centred in, 160 by default;
            ``module``, the largest module to read it in, if any.
    """
    from PIL import Image
    weights = config["weights"]
    pixel, ocr = load(weights) if isinstance(weights, (str, os.PathLike))\
        else weights
    pixel.eval(), ocr.eval()
    source = BytesIO(jpeg) if isinstance(jpeg, bytes) else jpeg
    photo = Image.open(source).convert("RGB")
    if config.get("scale"):
        ratio = 24 / config["scale"]
        photo = photo.resize((round(photo.width * ratio),
                              round(photo.height * ratio)))
    size = config.get("size", 160)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(photo, ((size - photo.width) // 2,
                         (size - photo.height) // 2))
    image = np.asarray(canvas)
    classes = predict(pixel, image[None])[0]
    boxes, wires, names, types = read(ocr, image, classes)
    module = scan(classes, types.values())
    most = config.get("module") or MODULES[-1]
    if MODULES.index(module) > MODULES.index(most):
        raise ValueError(f"The picture shows a {module} diagram.")
    return decode_least(
        refine(classes, boxes, wires, names, types), module, most)


def flood(image: np.ndarray, threshold=128) -> np.ndarray:
    """
    The baseline classes of an image, see :data:`picture.COARSE`: a flood
    fill from its border through the light pixels is the background, the
    light pixels it does not reach are inside a box, the dark pixels wires.
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
    classes = np.zeros(light.shape, dtype=np.uint8)
    classes[~light] = COARSE.index("wire")
    classes[light & ~reached] = COARSE.index("box")
    return classes


def recovered(classes, boxes, wires, names, types, module, diagram) -> bool:
    """
    Whether the classes of a picture's pixels, its components and the
    names and types read on them decode to a diagram of a module, up to its
    axioms: the :func:`picture.scan` of the classes must be that module or a
    smaller one, a diagram of which is one of that module.
    """
    try:
        if MODULES.index(picture.scan(classes)) > MODULES.index(module):
            return False
        mask = picture.refine(classes, boxes, wires, names, types)
        return picture.equal(picture.decode(mask, module), diagram, module)
    except (AxiomError, ValueError, IndexError, NotImplementedError,
            AssertionError):
        return False


def score(args) -> bool:
    """ :func:`recovered` for a worker. """
    return recovered(*args)


def distance(a: str, b: str) -> int:
    """
    The edit distance of two strings.

    >>> distance("a.r", "a"), distance("f", "g")
    (2, 1)
    """
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row, previous = [i], row
        for j, y in enumerate(b, 1):
            row.append(min(previous[j] + 1, row[j - 1] + 1,
                           previous[j - 1] + (x != y)))
    return row[-1]


def text_accuracy(ocr: OCR, crops, strings) -> dict:
    """ The accuracy of the OCR model on crops: the fraction of strings it
    reads exactly, and of characters, one minus the edit distance over the
    length of the truth. """
    found = []
    for i in range(0, len(crops), 512):
        found += ocr.read(torch.from_numpy(crops[i:i + 512]).unsqueeze(1))
    return {
        "strings": float(np.mean([f == s for f, s in zip(found, strings)])),
        "characters": 1 - sum(map(distance, found, strings)) / sum(
            map(len, strings))}


def evaluate(data: dict, classes: np.ndarray, ocr: OCR, workers=4) -> dict:
    """
    On a :func:`dataset` and the predicted classes of its pixels: the exact
    recovery per module with the OCR model reading the predicted components,
    the same with :func:`flood` for classes, and with the true classes, names
    and types; the accuracy of the pixels per class, of the OCR model on the
    true crops of boxes and labels, and of the scan.
    """
    modules = [MODULES[m] for m in data["modules"]]
    runs = {"exact": [], "baseline": [], "oracle": []}
    for image, mask, guess, module, diagram in zip(
            data["images"], data["masks"], classes, modules,
            data["diagrams"]):
        runs["exact"].append(
            (guess, *read(ocr, image, guess), module, diagram))
        base = flood(image)
        runs["baseline"].append(
            (base, *read(ocr, image, base), module, diagram))
        truth, boxes, wires, labels, names, reads = truths(mask)
        attached = picture.attach(labels, wires)
        types = {attached[t]: s for t, s in reads.items() if t in attached}
        runs["oracle"].append(
            (truth, boxes, wires, names, types, module, diagram))
    result = {}
    with ProcessPoolExecutor(workers) as pool:
        for name, jobs in runs.items():
            hits, count = Counter(), Counter(modules)
            for module, ok in zip(
                    modules, pool.map(score, jobs, chunksize=16)):
                hits[module] += ok
            result[name] = {m: hits[m] / count[m] for m in MODULES
                            if count[m]}
    truth = picture.coarsen(data["masks"])
    result["pixels"] = {COARSE[c]: float((classes[truth == c] == c).mean())
                        for c in np.unique(truth)}
    result["ocr"] = {kind: text_accuracy(ocr, *found)
                     for kind, label in (("names", False), ("types", True))
                     if len((found := texts(data, label))[1])}
    result["scan"] = float(np.mean([
        picture.scan(c) == picture.scan(t) for c, t in zip(classes, truth)]))
    return result
