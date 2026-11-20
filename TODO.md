> can we simplify the from_callable away and make it just the type used as decorator?
>
> yeah type the domain, for Dim I’m not sure why it can’t return a Term instead of a Diagram? it would make sense if it’s already in that module, tensor.Dim takes a callable returns a tensor.Term
>
> new tensor.Term
>
> it should be possible to choose the spiders for a subclass but it’s true we can’t pick different spiders in the same diagram

- [x] tensor.Term: Dim called on a function is an Abstraction, terms over Dim with application by cups, abstraction by caps, copy and discard of the diagram class
- [x] test_Term and the doctest in discopy/tensor.py, a CHANGELOG bullet
- [x] rewrite the strict/sloppy notebook with tensor.Term, the copies as subclasses of tensor.Diagram
- [ ] run the gates and the notebook check
