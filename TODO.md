> new idea: let's use graphviz layout to generate synthetic data from combinatorial maps and hypergraphs
>
> the graphviz visualisation should respect the port order, you can use an html table with ports as columns
>
> Go with oriented graphs and open the PR with your other defaults

- [x] picture.samples(layout="graphviz"): hypergraphs laid out by dot, each box an HTML table with one column per port, rendered with exact masks
- [x] picture.graphs: named families (complete, star, cycle, cube, grid, tree) as oriented compact diagrams, each edge from its smaller vertex to its larger
- [x] tests, docs in from_picture notebook, changelog
