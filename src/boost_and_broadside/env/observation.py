from dataclasses import dataclass
from enum import StrEnum

import torch


class ObsKey(StrEnum):
    POS = "pos"
    VEL = "vel"
    ATT = "att"
    ANG_VEL = "ang_vel"
    HEALTH = "health"
    POWER = "power"
    COOLDOWN = "cooldown"
    TEAM_ID = "team_id"
    ALIVE = "alive"
    RADIUS = "radius"
    PREVIOUS_ACTION = "previous_action"


_LEGACY_KEY_MAP: dict[str, tuple[ObsKey, int]] = {
    "prev_power": (ObsKey.PREVIOUS_ACTION, 0),
    "prev_turn":  (ObsKey.PREVIOUS_ACTION, 1),
    "prev_shoot": (ObsKey.PREVIOUS_ACTION, 2),
}


@dataclass(frozen=True)
class MVPObservation:
    """Typed immutable observation for all entities.

    data: maps ObsKey → tensor of shape (B, N+M, ...) or (T, B, N+M, ...) etc.

    team_id:  (B, N+M)   int32 — 0/1 ships, 2 obstacles
    alive:    (B, N+M)   bool
    all others have a trailing feature dimension.
    """

    data: dict

    # ------------------------------------------------------------------
    # Key access — supports ObsKey enum or str (including legacy names)
    # ------------------------------------------------------------------

    def __getitem__(self, key) -> torch.Tensor:
        if isinstance(key, ObsKey):
            return self.data[key]
        if key in _LEGACY_KEY_MAP:
            obs_key, channel = _LEGACY_KEY_MAP[key]
            return self.data[obs_key][..., channel]
        return self.data[ObsKey(key)]

    def __contains__(self, key) -> bool:
        if isinstance(key, ObsKey):
            return key in self.data
        if key in _LEGACY_KEY_MAP:
            return True
        try:
            return ObsKey(key) in self.data
        except ValueError:
            return False

    def items(self):
        return self.data.items()

    # ------------------------------------------------------------------
    # Typed property accessors
    # ------------------------------------------------------------------

    @property
    def pos(self) -> torch.Tensor:
        return self.data[ObsKey.POS]

    @property
    def vel(self) -> torch.Tensor:
        return self.data[ObsKey.VEL]

    @property
    def att(self) -> torch.Tensor:
        return self.data[ObsKey.ATT]

    @property
    def ang_vel(self) -> torch.Tensor:
        return self.data[ObsKey.ANG_VEL]

    @property
    def health(self) -> torch.Tensor:
        return self.data[ObsKey.HEALTH]

    @property
    def power(self) -> torch.Tensor:
        return self.data[ObsKey.POWER]

    @property
    def cooldown(self) -> torch.Tensor:
        return self.data[ObsKey.COOLDOWN]

    @property
    def team_id(self) -> torch.Tensor:
        return self.data[ObsKey.TEAM_ID]

    @property
    def alive(self) -> torch.Tensor:
        return self.data[ObsKey.ALIVE]

    @property
    def radius(self) -> torch.Tensor:
        return self.data[ObsKey.RADIUS]

    @property
    def previous_action(self) -> torch.Tensor:
        return self.data[ObsKey.PREVIOUS_ACTION]

    # ------------------------------------------------------------------
    # Immutable update / structural ops
    # ------------------------------------------------------------------

    def update(self, key: ObsKey, value: torch.Tensor) -> "MVPObservation":
        new_data = dict(self.data)
        new_data[key] = value
        return MVPObservation(data=new_data)

    def flip_team(self, num_ships: int) -> "MVPObservation":
        """Swap team IDs 0 and 1 for the first num_ships entity slots."""
        team_id = self.data[ObsKey.TEAM_ID].clone()
        ship_slice = team_id[..., :num_ships]
        flipped = torch.where(
            ship_slice == 0, 1, torch.where(ship_slice == 1, 0, ship_slice)
        )
        team_id[..., :num_ships] = flipped
        return self.update(ObsKey.TEAM_ID, team_id)

    def slice_envs(self, idx) -> "MVPObservation":
        return MVPObservation(data={k: v[idx] for k, v in self.data.items()})

    def slice_time(self, start: int, end: int) -> "MVPObservation":
        return MVPObservation(data={k: v[start:end] for k, v in self.data.items()})

    def concat_batch(self, other: "MVPObservation") -> "MVPObservation":
        """Concatenate two observations along the batch (env) dimension (dim 0)."""
        return MVPObservation(data={
            k: torch.cat([v, other.data[k]], dim=0) for k, v in self.data.items()
        })

    # ------------------------------------------------------------------
    # Construction from legacy dict
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(d: dict) -> "MVPObservation":
        """Build from a raw obs dict, handling legacy key names.

        Recognises 'prev_power'/'prev_turn'/'prev_shoot' (old three-key layout)
        and combines them into PREVIOUS_ACTION: (B, N+M, 3).
        """
        data: dict[ObsKey, torch.Tensor] = {}
        for k, v in d.items():
            try:
                data[ObsKey(k)] = v
            except ValueError:
                pass

        if ObsKey.PREVIOUS_ACTION not in data:
            if "prev_power" in d:
                p = d["prev_power"]
                t = d["prev_turn"]
                s = d["prev_shoot"]
                data[ObsKey.PREVIOUS_ACTION] = torch.stack([p, t, s], dim=-1)

        return MVPObservation(data=data)
