> "next#524: Encode layer boundaries — encode writes where a layer ends so the round trip is the identity; to_staircases flattens explicitly." and "next#524 return shape: B: nested lists — (dom, [[(box, offset), ...], ...]) one list per layer."

- [x] encode returns (dom, one list of (box, offset) per layer); decode reads it
- [x] to_staircases flattens explicitly; update every caller
- [x] the issue's reproducer as a test; changelog
