"""How the solver says no (ship-card spec 4)."""


class Infeasible(ValueError):
    """A handle set no card can meet, with the reason."""


class RedundantHandles(UserWarning):
    """Two handles are nearly redundant: the solve is ill-conditioned."""
