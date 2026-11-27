`csp.Language.propagate`, arc consistency: a set of candidates per
variable, kept to the values each constraint supports, until no set
shrinks or for a number of rounds. It removes no value a solution takes,
and when the incidence graph is a tree its fixed point is the projection
of the solutions (Freuder, JACM 1982) ([next#558](https://github.com/rel-int/discopy-next/pull/558)).
