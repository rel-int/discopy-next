> Review round on #554 (chatgpt-codex-connector[bot]), head e5679904:
>
> 1. discopy/tensor.py:768 (thread 4588275702)
> **  Preserve shared diagonal indices when contracting with Quimb**
>
> When a spider connects two ports of one box and also another box or boundary port, the new operands repeat that spider's index within an array and share it with another array. However, `Quimb.network` in `discopy/matrix.py` removes and sums every locally repeated index, assuming it is a closed trace. For example, the hypergraph in the new `spider_operands` docstring, with its box mapped to `diag(1, 2)`, should evaluate to `[1, 2]`, but using `contractor=Quimb()` or contracting `F.to_quimb(graph)` produces `[3, 3]`. Update the Quimb adapter to retain one diagonal axis for repeated indices that occur elsewhere in the network instead of summing them prematurely.
>
> Useful? React with 👍 / 👎.
>
> 2. discopy/tensor.py:672 (thread 4588275732)
> **  Restrict spider operands to Frobenius hypergraphs**
>
> For non-Frobenius hypergraphs, this unconditional dispatch bypasses the category check and diagram fallback in `__call__`, so the newly accepted `F.contract(graph)` and `F.to_quimb(graph)` paths can silently change tensor semantics. For example, let `graph = compact.Cup(x, x.r).to_hypergraph()` and map `x` to `Dim(2, 3)`: `F(graph)` correctly pairs the component dimensions in reverse order, whereas `spider_operands` flattens both boundary ports to dimension six and pairs them with an ordinary identity, producing different entries after reshaping to `(2, 3, 3, 2)`. Apply the same Frobenius-category guard here and translate other hypergraphs through their diagram and combinatorial map.
>
> Useful? React with 👍 / 👎.

- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 16:15 discopy/tensor.py:768 ** Preserve shared diagonal indices when contracting with Quimb** When a spider connects two ports of one box…
- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 16:15 discopy/tensor.py:672 ** Restrict spider operands to Frobenius hypergraphs** For non-Frobenius hypergraphs, this unconditional disp…
