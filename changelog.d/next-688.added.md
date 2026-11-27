`discopy.drawing.optimiser`, work in progress: a layout apart from
`Diagram.draw`, which is unchanged: a monoidal functor from diagrams to `OpenQP`, the category
of open quadratic programs over the abscissas of a drawing, where
composition identifies the boundary variables and adds the costs and
tensor adds a separation constraint. `optimiser.to_drawing` solves it with
OSQP, the optional extra `optimiser`, keeping the isotopy class and
weighing the straightness of the wires against the width of the drawing
and of its boxes, and returns an `optimiser.Optimised` drawing whose wires
go through waypoints ([next#688](https://github.com/rel-int/discopy-next/pull/688)). `optimiser.to_pivotal_drawing` draws a
pivotal diagram as a planar map with a fixed rotation system, its boxes
rotating and its wires free, up to isotopy: a box has a centre and a
rotation, cups and caps are turning points of a wire, and each step of the
descent from the monoidal layout is a quadratic program, kept only when
the drawing stays valid ([next#689](https://github.com/rel-int/discopy-next/pull/689)).
