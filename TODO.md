> Request from toumix, relayed from his Puck thread https://ampcode.com/threads/T-01a1fdf5-48e7-72a1-9471-54dd9de19cf2: work on diagram drawing in general, as the inverse of recognition: diagrams to pictures, in rel-int/discopy-next.
>
> Core idea from toumix: a layout algorithm for each level of the hierarchy that respects the isotopy of diagrams and optimises an aesthetic criterion, such as compactness and wire straightness. It should be implemented as monoidal functors into a category of open constrained optimisation problems: composition identifies boundary variables and adds costs, tensor adds separation constraints.
>
> Suggested first milestone: a functor from monoidal.Diagram to an OpenQP category, x-coordinates only, with a straightness and width cost, solved with cvxpy or OSQP, compared against the current drawing on the gallery examples. Design the interface so pivotal angles and rotation-system constraints plug in later.

- [x] discopy.drawing.layout: OpenQP, the category of open QPs over x-coordinates, with id, then, tensor
- [x] Diagram.draw(layout='qp'): the functor into OpenQP solved by OSQP, as a Drawing with waypoints
- [ ] Tests: isotopy kept (no crossing, no wire through a box, boundary order), narrower than today on the gallery
- [ ] Gallery comparison of today's layout against the QP layout, in the PR
- [ ] CHANGELOG, docs, checks
