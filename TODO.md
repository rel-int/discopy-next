> "next#524: Encode layer boundaries — encode writes where a layer ends so the round trip is the identity; to_staircases flattens explicitly." and "next#524 return shape: B: nested lists — (dom, [[(box, offset), ...], ...]) one list per layer."

- [WIP] @session_01MEJxy45CJtENv2bizg14LE-2026-11-18 07:29 encode returns (dom, one list of (box, offset) per layer); decode reads it
- [ ] to_staircases flattens explicitly; update every caller
- [ ] the issue's reproducer as a test; changelog
