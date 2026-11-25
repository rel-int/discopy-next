> another feature we never got time to implement: drawing diagrams from left to right. even better, drawing diagrams from left-to-right inside of a box that goes top-to-bottom, target is to reproduce this diagram from higher-order DisCoCat (screenshot: "Very big car", green word boxes with lambda terms, each holding a red slot with a diagram drawn left to right)

- [x] Backend: draw a nested drawing through the transpose map
- [x] Bubble: opt-in direction, drawn as one box with its args left to right inside
- [x] Top level: draw(direction=...)
- [x] Reproduce the Very big car figure as a doctest
- [x] Changelog, checks
- [x] Apply toumix's rulings on the two design asks of #728 (relative bool vs direction; inner wire labels), defaults due 2026-11-27
