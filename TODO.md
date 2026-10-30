# TODO

> Reading through the discopy-next docs I found some issues:
>
> * the architecture diagram that I wrote by hand in html should be updated with the new syntax modules
> * in the menu under syntax "discopy.spacial" is the only item that has the discopy prefix, in semantics it's only "discopy.kleisli" and "discopy.monad"
> * the docs for feedback and traced is a bit awkward with the axioms repeated
>
> another bug: broken link in the categorical deep learning notebook discopy.group points to https://docs.discopy.org/en/main/_api/discopy.group.html but there's no such module upstream, fix all the broken links of that shape using relative links instead

- [x] architecture diagram: spacial, comarkov, closed, cmap, ehypergraph
- [x] menu: import spacial, kleisli and monad in `discopy/__init__.py`
- [x] feedback: one section per law, the trace checked under it
- [x] notebooks: relative links to the API docs
