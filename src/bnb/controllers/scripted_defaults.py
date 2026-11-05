"""The scripted agent's default parameters per game mode (`MODE_DEFAULTS`), each the choice
of that mode's own parameter search, and each search's final population
(`scripted_populations.json`, the 32 sets of each mode's last generation, best first: kept
for later use, read by nothing here but `populations`). The section notes record the
searches."""

import json
import pathlib

from .scripted_parameters import MODES, ModeParameters, ScriptedParameters, from_row

POPULATIONS = pathlib.Path(__file__).with_name("scripted_populations.json")

MODE_DEFAULTS = ModeParameters({mode: ScriptedParameters() for mode in MODES})


def populations() -> dict[str, list[ScriptedParameters]]:
    """Each mode's final search population, best first."""
    stored = json.loads(POPULATIONS.read_text())
    return {mode: [from_row(row) for row in stored["populations"][mode]] for mode in MODES}
