`drawing.from_picture(jpeg, config)`: the diagram a picture shows, read
by `drawing.vision`, a pixel net (`PixelNet`) that classes each pixel and
a small OCR model (`OCR`) that reads box names and wire types, then decoded
in the least module it decodes in (`picture.decode_least`); with `train`,
`train_ocr` and `evaluate`, and `picture.crop` ([next#602](https://github.com/rel-int/discopy-next/pull/602)).
