The Typst backend draws its labels after every box and wire, as TikZ puts
them on a layer of their own, so that a box drawn after a wire label no
longer hides it ([next#648](https://github.com/rel-int/discopy-next/pull/648)).
