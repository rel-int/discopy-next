> cubic-dev-ai review on #688 (https://github.com/rel-int/discopy-next/pull/688#pullrequestreview-6091681017):
>
> > discopy/drawing/optimiser.py:1215 [P2] This loop creates waypoints at both ends of every piece, coincident with the boundary or box-port nodes. `Waypoints.draw_wires` renders the resulting edges as tangent-driven Beziers, adding unwanted excursions at wire endpoints; create waypoints only where the curve actually needs intermediate nodes.

- [ ] optimiser.py:1215 no excursion at a waypoint on its end
