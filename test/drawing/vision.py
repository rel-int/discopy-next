from io import BytesIO

import numpy as np
from pytest import importorskip, raises

from discopy.drawing import from_picture, picture

torch = importorskip("torch")
from discopy.drawing import vision  # noqa: E402


class TruePixels(torch.nn.Module):
    """ A pixel net that knows the classes of the one picture it reads. """
    def __init__(self, mask):
        super().__init__()
        self.classes = torch.from_numpy(picture.coarsen(mask))

    def forward(self, x):
        scores = torch.nn.functional.one_hot(self.classes, len(picture.COARSE))
        return scores.permute(2, 0, 1)[None].float().expand(len(x), -1, -1, -1)


class TrueText:
    """ An OCR model that reads, from each crop, the text the true mask
    puts there: the crops come in the order :func:`vision.read` cuts them. """
    def __init__(self, texts):
        self.texts = texts

    def eval(self):
        return self

    def read(self, crops):
        return self.texts[:len(crops)]


def jpeg(image):
    from PIL import Image
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, "PNG")
    return buffer.getvalue()


def test_from_picture_reads_a_picture_back():
    from discopy.symmetric import Ty, Box, Swap
    x, y = Ty('a'), Ty('b')
    diagram = Box('f', x, x @ y, shape="triangle") >> Swap(x, y)
    image, mask = picture.render(diagram, 160), picture.render(
        diagram, 160, mask=True)
    classes = picture.coarsen(mask)
    boxes, wires, labels = picture.regions(classes)
    names = [picture.LABELS[np.bincount(mask[boxes == b]).argmax()]
             for b in np.unique(boxes[boxes >= 0])]
    types = [picture.LABELS[np.bincount(mask[labels == t]).argmax()]
             .removeprefix("label ")
             for t in picture.attach(labels, wires)]
    weights = (TruePixels(mask), TrueText(names + types))
    result = from_picture(jpeg(image), {"weights": weights})
    assert picture.equal(result, diagram, "symmetric")
    with raises(ValueError):
        from_picture(jpeg(image), {"weights": weights, "module": "monoidal"})


def test_save_and_load(tmp_path):
    pixel, ocr = vision.PixelNet(4, 2), vision.OCR(8)
    vision.save(tmp_path / "weights.pt", pixel, ocr)
    loaded, read = vision.load(tmp_path / "weights.pt")
    assert all(torch.equal(a, b) for a, b in zip(
        pixel.state_dict().values(), loaded.state_dict().values()))
    assert read.rnn.hidden_size == 8


def test_train_ocr_lowers_its_loss():
    """ #653: `train_ocr` seeds torch for the training it does, which is
    after a network it is given has drawn its weights, so the test seeds
    for the one it builds: the loss the training starts from otherwise
    comes from whatever ran before it on the xdist worker, and the ratio
    this asserts is a draw rather than a measurement. """
    data = vision.dataset(4, seed=600, modules=("monoidal", ), workers=1)
    crops, strings = vision.texts(data, True)
    losses = []
    torch.manual_seed(0)
    vision.train_ocr(crops, strings, epochs=10, batch=8, ocr=vision.OCR(8),
                     log=lambda line: losses.append(float(line.split()[-1])))
    assert losses[-1] < .8 * losses[0]


def test_flood_is_background_box_and_wire():
    from discopy.monoidal import Ty, Box
    image = picture.render(Box('f', Ty('a'), Ty('a')), 64)
    classes = vision.flood(image)
    assert {picture.COARSE[c] for c in np.unique(classes)} == {
        "background", "wire", "box"}


def test_distance():
    assert vision.distance("", "abc") == 3
    assert vision.distance("kitten", "sitting") == 3


def test_evaluate_on_one_module():
    diagram, image, mask = picture.sample("monoidal", seed=0)
    data = {"images": image[None], "masks": mask[None],
            "modules": np.array([0]), "diagrams": [diagram]}
    classes = picture.coarsen(mask)[None]
    result = vision.evaluate(data, classes, vision.OCR(), workers=1)
    assert set(result["oracle"]) == {"monoidal"}
    assert result["oracle"]["monoidal"] == 1


def test_from_picture_takes_a_path_to_weights(tmp_path, monkeypatch):
    from discopy.monoidal import Ty, Box
    diagram = Box('f', Ty('a'), Ty('a'))
    image, mask = picture.render(diagram, 160), picture.render(
        diagram, 160, mask=True)
    path, loaded = tmp_path / "weights.pt", []
    monkeypatch.setattr(vision, "load", lambda p: loaded.append(p) or (
        TruePixels(mask), TrueText(["f", "a", "a"])))
    result = from_picture(jpeg(image), {"weights": path, "module": None})
    assert loaded == [path] and picture.equal(result, diagram, "monoidal")


def test_from_picture_keeps_to_its_module():
    from discopy import frobenius
    from discopy.utils import AxiomError
    x = frobenius.Ty('a')
    diagram = frobenius.Cup(x, x)
    image, mask = picture.render(diagram, 160), picture.render(
        diagram, 160, mask=True)
    classes = picture.coarsen(mask)
    _, wires, labels = picture.regions(classes)
    types = [picture.LABELS[np.bincount(mask[labels == t]).argmax()]
             .removeprefix("label ")
             for t in picture.attach(labels, wires)]
    weights = (TruePixels(mask), TrueText(types))
    assert picture.equal(
        from_picture(jpeg(image), {"weights": weights}), diagram, "frobenius")
    with raises((ValueError, AxiomError)):
        from_picture(jpeg(image), {"weights": weights, "module": "compact"})


def test_train_ocr_trains_a_loaded_model():
    crops = np.random.default_rng(0).random((8, *picture.CROP))
    ocr = vision.OCR().eval()
    before = ocr.convs[1].running_mean.clone()
    vision.train_ocr(crops.astype(np.float32), ["f"] * 8, epochs=1,
                     batch=8, ocr=ocr, log=lambda _: None)
    assert not torch.equal(before, ocr.convs[1].running_mean)


def test_evaluate_without_wire_labels():
    diagram, image, mask = picture.sample(
        "monoidal", seed=0, wire_labels=False)
    data = {"images": image[None], "masks": mask[None],
            "modules": np.array([0]), "diagrams": [diagram]}
    result = vision.evaluate(
        data, picture.coarsen(mask)[None], vision.OCR(), workers=1)
    assert "types" not in result["ocr"]


def test_train_rejects_a_batch_larger_than_the_data():
    diagram, image, mask = picture.sample("monoidal", seed=0, size=64)
    data = {"images": image[None], "masks": mask[None],
            "modules": np.array([0]), "diagrams": [diagram]}
    with raises(ValueError):
        vision.train(data, epochs=1, batch=2, log=lambda _: None)


def test_decode_least_rejects_an_empty_range():
    mask = picture.render(picture.load("monoidal").Box(
        'f', picture.load("monoidal").Ty('a'),
        picture.load("monoidal").Ty('a')), 96, mask=True)
    with raises(ValueError):
        picture.decode_least(mask, "compact", "monoidal")
