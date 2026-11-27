optyx's `PermanentBackend` evaluates a sum of diagrams whose terms carry
different numbers of photons, padding each term's amplitudes with zeros,
where it used to add them by broadcasting or raise ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
