`optyx.utils.perceval_conversion.ff_configurator` gives each symbol of a
configurator's controlled circuit the value its own name is given in the
configuration: it used to zip `U.free_symbols`, a set, with the values of
the configuration dict, so which symbol got which value depended on
`PYTHONHASHSEED` and the circuit a feed-forward configuration converted to
was silently another configuration's. A configuration that gives no value
for a symbol now raises `ValueError` rather than pairing that symbol with
whatever came next. The dead `use_provider_dom` flag goes with it: a
parameter threaded through `_feedforward_common` that nothing read, its
branch commented out, so the signature promised a provider domain the code
never computed ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
