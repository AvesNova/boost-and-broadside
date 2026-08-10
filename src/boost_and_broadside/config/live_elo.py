"""The approximate live Elo gauge training rates itself on.

Training needs cheap stationary references long before the live policy can beat
the scripted controller, so the in-training ladder is *defined* rather than
measured:

    uniform random        live Elo 0
    scripted controller   live Elo 1000
    semi-random at ``p``  live Elo 1000·p

Above scripted the ordinary live update continues on the same numbers.

The linear rung assignment is deliberately rough at the weak end and accurate
enough near scripted. Against ratings fitted by ``bnb semi-random`` and
regauged to the same anchors:

    p     rl fitted   fields fitted   linear    Δ rl     Δ fields
    0.2        93.8           122.7      200   −106.2       −77.3
    0.3       196.3           219.3      300   −103.7       −80.7
    0.4       351.1           348.9      400    −48.9       −51.1
    0.5       465.3           481.8      500    −34.7       −18.2
    0.6       604.9           604.6      600     +4.9        +4.6
    0.7       698.8           716.0      700     −1.2       +16.0
    0.8       804.2           797.5      800     +4.2        −2.5
    0.9       898.3           930.7      900     −1.7       +30.7
    0.95      957.8           987.2      950     +7.8       +37.2

Two consequences are accepted along with the approximation. Opponent sampling
is proximity-weighted, so a rung placed ~100 points high early draws league and
evaluation games slightly sooner than a fitted rung would. And the ladder
snapshot grid (``TrainConfig.elo_milestone_gap``) is read on these numbers, so
which updates freeze a checkpoint moves with them.

What this buys is that the gauge no longer depends on an offline fit that has
to be re-run — and silently invalidates every rung when it is not — each time
the environment moves. It is the same defined scale in every environment, which
also means it makes no claim about the two being equally hard.

This is a training instrument: it steers opponent selection, the live progress
curve, and milestone placement. It is not a calibrated rating and must never be
published as one. Post-hoc ``bnb elo-calibrate`` fits actual match outcomes and
is the source of every reported number; the two are named ``live_elo`` and
``calibrated_elo`` throughout the code, logs, and artifacts, and are never
substituted for one another. ``bnb semi-random`` measures how far this
approximation sits from a fitted gauge — it validates the ladder rather than
producing it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# The gauge's two defining points. Random is its zero and the scripted
# controller its unit; neither is a measurement, and neither moves during a run.
LIVE_RANDOM_ELO = 0.0
LIVE_SCRIPTED_ELO = 1000.0


def validate_live_reference_probabilities(probabilities: Sequence[float]) -> None:
    """Raise unless the rungs are strictly increasing scripted-action odds.

    ``0`` is the random agent and ``1`` is the scripted controller; both are
    already on the ladder, so an interior rung must lie strictly between them.
    """

    previous = 0.0
    for probability in probabilities:
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            raise ValueError(
                f"live reference probabilities must be numbers, got {probability!r}"
            )
        if not math.isfinite(probability) or not previous < probability < 1.0:
            raise ValueError(
                "live reference probabilities must be finite and strictly increasing "
                f"in (0, 1), got {tuple(probabilities)!r}"
            )
        previous = float(probability)


def live_reference_elo(
    p_scripted: float,
    *,
    scripted_elo: float = LIVE_SCRIPTED_ELO,
) -> float:
    """Return the live rating assigned to a semi-random rung at ``p_scripted``.

    Linear between the gauge's two defining points, which is the whole content
    of the approximation: ``LIVE_RANDOM_ELO`` at 0 and ``scripted_elo`` at 1.
    """

    if not math.isfinite(p_scripted) or not 0.0 <= p_scripted <= 1.0:
        raise ValueError(f"p_scripted must be finite and in [0, 1], got {p_scripted}")
    if not math.isfinite(scripted_elo):
        raise ValueError(f"scripted_elo must be finite, got {scripted_elo}")
    return LIVE_RANDOM_ELO + (scripted_elo - LIVE_RANDOM_ELO) * float(p_scripted)


def live_reference_ladder(
    probabilities: Iterable[float],
    *,
    scripted_elo: float = LIVE_SCRIPTED_ELO,
) -> tuple[tuple[float, float], ...]:
    """Return the ``(p_scripted, live_elo)`` rungs for a profile's probabilities.

    One derivation site, so the roster the trainer registers and the ladder a
    resolved configuration reports can never disagree.
    """

    rungs = tuple(float(probability) for probability in probabilities)
    validate_live_reference_probabilities(rungs)
    return tuple(
        (probability, live_reference_elo(probability, scripted_elo=scripted_elo))
        for probability in rungs
    )
