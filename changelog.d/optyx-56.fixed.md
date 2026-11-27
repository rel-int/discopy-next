An array-backed `discopy.optyx` box reads its size off the array's shape,
so the array can be a PyTorch tensor that requires grad ([optyx#56](https://github.com/rel-int/optyx/pull/56)).
