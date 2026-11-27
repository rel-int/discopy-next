`drawing.picture`, pictures of diagrams as training data for reading them
back: `samples` draws random diagrams of `monoidal`, `symmetric`,
`compact` or `frobenius` one generator at a time, renders each with a mask
labelling every pixel by its wire type, box name or structural primitive,
by hand with `xkcd`; `augment` warps, rotates and changes the perspective
of the image and its mask together, then restyles the image alone (chalk,
whiteboard, blur, JPEG); `draw(backend=...)` takes the backend to draw with
([next#573](https://github.com/rel-int/discopy-next/pull/573)).
