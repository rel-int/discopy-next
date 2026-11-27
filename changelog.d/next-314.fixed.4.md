`optyx.photonic.Select()` and `Create()` default to one photon on one mode,
as the `zw` boxes they wrap do, so that their `photons` and the path matrix
they build agree with the `qmode` they are typed on ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
