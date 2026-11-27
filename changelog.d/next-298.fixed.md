The SEMM compiler of `discopy.optyx` on a path cover of more than one path,
which used to recurse forever: `compiler.mbqc.add_fusion_order` closes the
past of a node without recursion, so a node fused to something in its own
past comes back; `GFlow.partial_order` leaves two nodes of one layer
incomparable, as a foliation does, rather than each in the other's past;
`compile_to_semm` extends the order to the breaks that join the paths,
which are nodes of the resource state and not of the graph; and
`get_creation_times` and `get_measurement_times` are keyed by node ID
rather than by position in the path, counting no measurement photon for an
output, so that the fusion photons pair up. What the instructions should
say about a break, so that a decompile reads the graph back, is open
([next#298](https://github.com/rel-int/discopy-next/issues/298)).
