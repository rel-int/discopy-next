`neural.network.Learner`, the parametric lenses of functions on tensors,
i.e. a learnable network with its weights and their gradients as a pair
of parameters: `Learner.reverse` is the reverse derivative of a parametric
network, `Lens.descent` gradient descent as a lens and `Learner.descent`
the reparametrisation by it, one step of an optimiser, while a
reparametrisation by `Lens.copy` ties weights and sums their gradients.
`neural.torch.Module` takes a parametric network with its initial weights
and holds them as its parameters, one per leg of the parameter space
([next#517](https://github.com/rel-int/discopy-next/pull/517)).
