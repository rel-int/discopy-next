`neural.torch.Module` now survives `copy.deepcopy` and `torch.save`: a copy
computes with and backpropagates into its own parameters, so `AveragedModel`
(SWA/EMA) and `best = copy.deepcopy(model)` work ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
