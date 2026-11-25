> USER feedback while reading the discopy-next docs. Implement it on a new branch and open a pull request. Do the work yourself; do not create another thread.
>
> 1. The architecture diagram included in the discopy.abc docstring (docs/api/architecture.html, rendered at _api/discopy.abc.html) is far from exhaustive, and adding everything would look too messy. Keep it only about syntax: drop everything semantics, lambek, and quantum. The syntax box stays. Update the surrounding prose in discopy/abc.py if it describes the dropped parts. Leave the unused CSS classes in docs/_style/custom.css; they do no harm.
>
> 2. MonoidalCategory.split_memory and MonoidalCategory.memory_size make no sense on MonoidalCategory. Move them to FeedbackCategory.
>
> 3. keyword_form is too boring to have its place in the abc autosummary (the Functions admonition in the discopy.abc module docstring). Make it a method of FeedbackCategory instead of a module-level function, and drop that Functions admonition if nothing else remains in it.
>
> The wrapping exists so that a trace defined in the positional form (n, left) also takes the keyword form (dom, cod, mem) of FeedbackCategory.feedback. MonoidalCategory.__init_subclass__ currently wraps any subclass that defines trace, including classes that are not feedback categories (discopy.matrix.Matrix, discopy.hypergraph.Hypergraph). After the move, that wrapping still has to happen for those classes. Put the hook where it still fires for them, and keep the doctest that keyword_form(Matrix.trace) is Matrix.trace working against the method. Update every caller, docstring cross-reference (:func:`keyword_form`, :meth:`MonoidalCategory.split_memory`, :meth:`MonoidalCategory.memory_size`), and test. Keep the public behaviour of trace and feedback the same.
>
> Follow the repo's AGENTS.md and changelog convention. Reply to https://ampcode.com/threads/T-01a2123a-e6ca-715d-8fa5-ea931602b4f3 when the pull request is open, with its URL, or with what blocked you.

- [x] Architecture diagram shows the syntax box only, prose updated
- [x] split_memory and memory_size move to FeedbackCategory
- [x] keyword_form becomes a FeedbackCategory method, Functions admonition dropped, callers and tests updated
- [x] CHANGELOG bullet and next-checks gates
- [x] egraph listed under semantics in the docs, dropped from the diagram (follow-up point from USER)
