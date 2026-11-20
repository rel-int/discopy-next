---
title: From Picture
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Reading a diagram back from its picture

`discopy.drawing.from_picture` reads a picture of a string diagram back into
a `Diagram`. Drawing is a functor from diagrams to images; reading is a
section of it, up to the axioms of the diagram's module.

The reading comes in two halves:

- **learned** (`drawing.vision`, PyTorch): a pixel net classes every pixel as
  background, wire, box, label, swap, cup, cap or spider, and a small OCR
  model reads the name of each box and the type of each labelled wire;
- **symbolic** (`drawing.picture`, no PyTorch): the classes are split into
  components, the names and types written back, the module chosen by a scan
  of the primitives that appear, and the diagram decoded.

Training data is free: `picture.samples` draws random diagrams of a module,
their pictures, and a mask that labels every pixel.

```python {.marimo}
import numpy as np
from discopy.drawing import picture

diagram, image, mask = picture.sample("compact", seed=6)
diagram
```

The picture, and its mask coarsened to the classes the pixel net predicts:

```python {.marimo}
import matplotlib.pyplot as plt

classes = picture.coarsen(mask)
figure, (left, right) = plt.subplots(1, 2, figsize=(8, 4))
left.imshow(image); left.set_title("picture"); left.axis("off")
right.imshow(classes, cmap="tab10", vmin=0, vmax=9)
right.set_title("classes"); right.axis("off")
figure
```

## Graphviz layouts

`samples(..., graphviz=True)` lays a diagram out by graphviz's `dot`
instead: each box is an HTML table with one column per port, so each wire
leaves its box where its port is, and the mask stays exact. With
`graphs=True` the diagrams are named graphs (complete, star, cycle, cube,
grid and tree), each edge oriented from its smaller vertex to its larger.

```python {.marimo}
from itertools import islice

laid_out = [image for _, image, _ in islice(picture.samples(
    "symmetric", seed=3, graphviz=True, graphs=True, xkcd=True), 3)]
figure_graphs, axes = plt.subplots(1, 3, figsize=(9, 3))
for axis, laid_out_image in zip(axes, laid_out):
    axis.imshow(laid_out_image); axis.axis("off")
figure_graphs
```

## The symbolic half

Given the classes, and something to read each box's name and each label's
type, `picture.read` gives the diagram back. Here the names and types are
read off the true mask, by majority: this is the ceiling of the reading,
what it gets with a perfect pixel net and a perfect OCR.

```python {.marimo}
from discopy.drawing.picture import LABELS, read, equal


def majority(rows, cols):
    return LABELS[np.bincount(mask[rows, cols]).argmax()]


read_back = read(classes, majority,
                 lambda rows, cols: majority(rows, cols).removeprefix("label "))
assert equal(read_back, diagram, "compact")
read_back
```

```python {.marimo}
picture.scan(classes)
```

## Numbers

Exact recovery is the fraction of pictures whose reading equals the drawn
diagram up to the axioms of its module, on 1,000 held-out pictures per
module, read with `from_picture`'s networks at 160 px (#559, run 2,
measured at 852c2c1):

| | monoidal | symmetric | compact | frobenius |
|---|---|---|---|---|
| exact recovery | 89.7% | 90.4% | 87.2% | 90.4% |
| flood-fill baseline | 3.5% | 8.2% | 0% | 0% |
| ceiling, true classes | 100% | 99.7% | 99.8% | 98.9% |
| true masks read back (seed 7) | 1000 | 999 | 996 | 990 |

The pixel net classes 96 to 99.9% of the pixels of each class right, the OCR
reads 95.7% of box names and 99.7% of wire types, and the scan picks the
right module for 99.8% of the pictures. `from_picture` takes 0.11 s a picture
on a CPU once loaded, and 1.7 s to import.

Run 7 trained on pictures half clean and half in whiteboard style (warped,
blurred, in perspective, hand-drawn), at 12 to 28 px per unit in a 320 px
frame, and was refuted on both its pre-registered lines (#559):

| exact recovery | monoidal | symmetric | compact | frobenius |
|---|---|---|---|---|
| whiteboard style, 6 to 10 px per unit | 7.4% | 13.7% | 2.8% | 5.4% |
| read from its warped true masks | 6.7% | 16.0% | 0.5% | 1.1% |
| the clean set above | 31.6% | 39.9% | 36.9% | 36.3% |

The second row is not a ceiling: a warped mask is resampled pixel by
pixel, which breaks the thin wires of a small picture, and the pixel net
reads three of the four modules better than these masks. On the clean
set the pixel net held, at 95 to 99.7% per class, and the OCR fell: it reads
74.9% of box names, from 95.7%.

Run 8 kept run 7's data and changed what the pixel net predicts: the
outline of a box instead of its fill, no cups and caps, and a `port` class
where a wire meets a box. A pre-pass then fills each closed outline and finds
cups and caps at the lowest and highest points of the wires. On true masks
this pre-pass reads back exactly as many pictures as before, at both scales.
The OCR was also given crops scaled to the text. It was refuted on both its
pre-registered lines (#559):

| exact recovery | monoidal | symmetric | compact | frobenius |
|---|---|---|---|---|
| the clean set above | 56.3% | 65.6% | 74.5% | 80.0% |
| whiteboard style, 12 to 20 px per unit | 7.7% | 15.1% | 5.2% | 9.3% |
| ceiling, clean true masks at 12 to 20 px per unit | 86.6% | 87.3% | 86.4% | 85.6% |

It beats run 7 on every module, by 25 to 44 points on the clean set, but
stays below the weights trained on clean pictures alone. The OCR still reads
73.4% of box names on the clean set, scaled crops or not. The third row is a
ceiling: at 12 px per unit a short wire has no room for its label, so its type
is not in the picture.

The weights are not in git. With weights `vision.save` wrote:

```py
from discopy.drawing import from_picture

from_picture(open("diagram.jpg", "rb").read(), {"weights": "weights.pt"})
```
