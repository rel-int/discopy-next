A box is an operation symbol of arity n: `cat.Box` takes `args`, the arrows
inside it, and the `method` a functor calls on their images, a generator
being a box with no args. The `Bubble` class of every module fuses into its
`Box` and `bubble_factory` goes: `f.bubble(...)` is `Box.from_args(f, ...)` (an
`Elementwise` in `tensor` and `quantum`),
a box with `args=(f, )`, `isinstance(f, Bubble)` becomes `f.args`, and
`Bubble` stays an alias of `Box.from_args`, or of `Elementwise.from_args` in
`tensor`, `quantum.circuit` and `quantum.zx`, and a tree dumped with a
`Bubble` factory still loads. The function a tensor bubble applies
elementwise moves to `tensor.Elementwise`, the box `tensor.Diagram.bubble`
builds, and a plain tensor box carries none;
`biclosed.Application.func` and `args` are spelled out as `function` and
`argument`, in its trees too, and a tree with the old keys still loads
([next#729](https://github.com/rel-int/discopy-next/pull/729)).
