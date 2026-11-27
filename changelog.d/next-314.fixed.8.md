`optyx.core.control.ControlledPhaseShift` applies the phase its function
reads off two or more control modes: it indexed its array along the first
axis only, with `array[i, :]` for a whole control configuration `i`, so the
per-configuration cell had the wrong shape and the reshape raised
([next#314](https://github.com/rel-int/discopy-next/issues/314)).
