`test/drawing/vision.py::test_train_ocr_lowers_its_loss` seeds torch before it
builds the `OCR` it trains: `train_ocr` seeds for its own shuffling, after a
network it is given has drawn its weights, so the loss the training started
from came from whatever ran before the test on its xdist worker and the ratio
it asserts ranged 0.56 to 0.81 ([next#656](https://github.com/rel-int/discopy-next/pull/656)).
