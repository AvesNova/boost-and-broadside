"""Offline audit of the next-state aux head of a running BC checkpoint.

Measures, per (ally|enemy) x (visible|belief) cell:
  * physical-unit one-step error of the head
  * the same for two baselines: identity (nothing changed) and, for position,
    linear motion (believed pos + believed vel * dt)
  * target-space per-dimension MSE for head / identity / linear
  * per-harmonic resultant length of the predicted Fourier moments
  * head-reported sigma vs realised residual (calibration)
  * hidden-age buckets for the belief cells

Nothing here writes to the project; results land in JSON next to this file.
"""

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, "/home/vizia/avesnova/boost-and-broadside/src")

from boost_and_broadside.config import EnvConfig, FrontlineConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.evaluation.agents import resolve_agent_spec
from boost_and_broadside.evaluation.environment import create_evaluation_env
from boost_and_broadside.evaluation.match import MatchRunner

RUN_DIR = Path("/home/vizia/avesnova/boost-and-broadside/checkpoints/bright-forest-747")
OUT = Path(__file__).with_name("aux_probe_result_dr.json")

NUM_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 48
NUM_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
WARMUP = int(sys.argv[3]) if len(sys.argv) > 3 else 100
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cuda"

AGE_EDGES = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float("inf")]


def load_configs():
    doc = json.loads((RUN_DIR / "config.json").read_text())
    cfg = doc["segments"][-1]["config"]
    ship = ShipConfig(**{**cfg["ship_config"], "world_size": tuple(cfg["ship_config"]["world_size"])})
    env_raw = dict(cfg["train_config"]["scales"][0]["env_config"])
    front = env_raw.pop("frontline")
    env = EnvConfig(**env_raw, frontline=FrontlineConfig(**front))
    model = ModelConfig(**cfg["model_config"])
    return ship, env, model


def latest_checkpoint() -> Path:
    steps = sorted(RUN_DIR.glob("step_*.pt"))
    return steps[-1]


class Accum:
    """Sum/sq-sum/count accumulators keyed by name, all on device."""

    def __init__(self, device):
        self.device = device
        self.sum: dict[str, torch.Tensor] = {}
        self.count: dict[str, torch.Tensor] = {}

    def add(self, key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        """values: (...,) or (..., D) aligned with mask (...,)."""
        m = mask.float()
        if values.dim() == mask.dim():
            total = (values * m).sum()
        else:
            total = (values * m.unsqueeze(-1)).sum(tuple(range(mask.dim())))
        prev = self.sum.get(key)
        self.sum[key] = total if prev is None else prev + total
        c = m.sum()
        prevc = self.count.get(key)
        self.count[key] = c if prevc is None else prevc + c

    def result(self) -> dict:
        out = {}
        for key, total in self.sum.items():
            count = float(self.count[key].item())
            value = total.cpu()
            mean = (value / max(count, 1.0)).tolist() if value.dim() else float(value.item() / max(count, 1.0))
            out[key] = {"mean": mean, "count": count}
        return out


def main() -> None:
    torch.manual_seed(0)
    dev = torch.device(DEVICE)
    ship, env_cfg, model_cfg = load_configs()
    ckpt = latest_checkpoint()
    N = env_cfg.num_ships
    print(f"checkpoint {ckpt.name}  envs={NUM_ENVS} steps={NUM_STEPS} device={DEVICE}", flush=True)

    agents = [
        resolve_agent_spec(str(ckpt), ship, model_cfg, DEVICE, str(RUN_DIR.parent), num_ships=N)
        for _ in range(2)
    ]
    coord = agents[0].agent.coordinator
    specs = {s.name: s for s in coord._predictor_specs}
    P = coord.total_prediction_dimension

    env = create_evaluation_env(NUM_ENVS, ship, env_cfg, dev)
    env.reset()
    runner = MatchRunner(
        env,
        agents,
        torch.zeros(NUM_ENVS, dtype=torch.long, device=dev),
        torch.ones(NUM_ENVS, dtype=torch.long, device=dev),
        ship,
        N,
    )
    runner.init_hidden()

    dt = ship.dt * env_cfg.action_repeat
    world = torch.tensor(ship.world_size, device=dev)
    acc = Accum(dev)
    # Last observed truth per ship slot, for a dead-reckoning baseline that uses
    # exactly the information the belief was seeded from.
    last_pos = torch.zeros(NUM_ENVS, N, 2, device=dev)
    last_vel = torch.zeros(NUM_ENVS, N, 2, device=dev)
    samples: dict[str, list[torch.Tensor]] = {}
    sample_cap = 400_000

    pos_x = specs["position_x"]
    pos_y = specs["position_y"]
    n_harm = pos_x.t_dim // 2
    att = specs["attitude"]
    n_att = att.t_dim // 2

    t0 = time.perf_counter()
    for step in range(NUM_STEPS):
        truth_now = observation_from_state(env.state, ship, include_bullets=runner.include_bullets)
        policy_obs = runner.observe()
        trace = runner.select_actions(policy_obs, trace_agents=frozenset({0}))
        view = trace.observations[0]
        pred = trace.predictions[0]
        alive_now = env.state.ship_alive[:, :N].clone()
        team = env.state.ship_team_id[:, :N]
        visible = view[ObsKey.VISIBLE][:, :N].bool()
        belief_valid = view[ObsKey.BELIEF_VALID][:, :N].bool()
        age = view[ObsKey.TIME_SINCE_OBSERVATION][:, :N, 0].float()
        curr_targets = coord.get_target_vector(view)[:, :N]
        seen = visible.unsqueeze(-1)
        last_pos = torch.where(seen, truth_now[ObsKey.POS][:, :N], last_pos)
        last_vel = torch.where(seen, truth_now[ObsKey.VEL][:, :N], last_vel)

        dones, truncated = runner.advance(trace.action)
        # TensorEnv never clears the spawn-reveal latch -- only YemongEnvWrapper
        # does, at the start of each decision -- so a plain MatchRunner loop
        # leaves every ship permanently revealed and measures a fog-free world.
        # Clear it the way the training wrapper does: this decision's respawns
        # only, with reset spawns re-latched by reset_finished below.
        env.state.ship_spawned.copy_(env.state.ship_respawned)
        done_any = dones | truncated
        truth_next = observation_from_state(env.state, ship, include_bullets=runner.include_bullets)
        alive_next = env.state.ship_alive[:, :N]

        if step >= WARMUP and pred is not None:
            with torch.no_grad():
                true_targets = coord.get_target_vector(truth_next)[:, :N]
                pred_targets = coord.apply_scaled_predictions(curr_targets, pred)

                cur = coord.decode_targets(curr_targets)
                # Linear-motion baseline for position, in target space.
                lin_x = cur["position_x"] + cur["velocity"][..., 0:1] * dt
                lin_y = cur["position_y"] + cur["velocity"][..., 1:2] * dt
                lin_targets = curr_targets.clone()
                lin_targets[..., pos_x.t_offset : pos_x.t_offset + pos_x.t_dim] = pos_x.target_encoder(lin_x)
                lin_targets[..., pos_y.t_offset : pos_y.t_offset + pos_y.t_dim] = pos_y.target_encoder(lin_y)
                # Attitude extrapolated by believed angular velocity (bonus baseline).
                att_ang = torch.atan2(cur["attitude"][..., 1:2], cur["attitude"][..., 0:1])
                att_lin = att_ang + cur["angular_velocity"] * dt
                lin_targets[..., att.t_offset : att.t_offset + att.t_dim] = att.target_encoder(
                    torch.cat([att_lin.cos(), att_lin.sin()], dim=-1)
                )

                labels = coord.compute_labels(curr_targets, true_targets)  # scaled
                scale = coord.label_scale_vector(dev)
                pred_mean = pred[..., :P].float()
                id_pred_scaled = coord.compute_labels(curr_targets, curr_targets)
                lin_pred_scaled = coord.compute_labels(curr_targets, lin_targets)

                variance = coord.prediction_variance(pred.float())  # (..., P)

                decoded = {
                    "head": coord.decode_targets(pred_targets),
                    "identity": cur,
                    "linear": coord.decode_targets(lin_targets),
                }
                truth = coord.decode_targets(true_targets)

                # Frontline respawn is instantaneous, so an alive-at-both-ends
                # test cannot see it: the ship is alive before and after, having
                # teleported to a spawn point in between. Training excludes those
                # transitions via ``transition_contiguous``; ``ship_respawned``
                # after the step is the same fact.
                respawned = env.state.ship_respawned[:, :N]
                base_ok = alive_now & alive_next & ~done_any.unsqueeze(-1) & ~respawned
                # Cells with "_anyalive" drop the alive-at-both-ends filter and
                # so reproduce what the training-time belief/* series measures.
                no_done = ~done_any.unsqueeze(-1).expand_as(alive_now)
                cells = {
                    "ally_visible": base_ok & (team == 0) & visible,
                    "ally_hidden": base_ok & (team == 0) & belief_valid & ~visible,
                    "enemy_visible": base_ok & (team == 1) & visible,
                    "enemy_hidden": base_ok & (team == 1) & belief_valid & ~visible,
                    "enemy_visible_anyalive": no_done & (team == 1) & visible,
                    "enemy_hidden_anyalive": no_done & (team == 1) & belief_valid & ~visible,
                    "ally_visible_anyalive": no_done & (team == 0) & visible,
                }

                def physical(d):
                    pos = torch.cat([d["position_x"], d["position_y"]], dim=-1)
                    tpos = torch.cat([truth["position_x"], truth["position_y"]], dim=-1)
                    delta = torch.remainder(pos - tpos + world / 2.0, world) - world / 2.0
                    a = torch.nn.functional.normalize(d["attitude"], dim=-1)
                    ta = torch.nn.functional.normalize(truth["attitude"], dim=-1)
                    return {
                        "position_px": delta.norm(dim=-1),
                        "velocity_px_s": (d["velocity"] - truth["velocity"]).norm(dim=-1),
                        "speed_px_s": (d["velocity"].norm(dim=-1) - truth["velocity"].norm(dim=-1)).abs(),
                        "attitude_rad": torch.acos((a * ta).sum(-1).clamp(-1, 1)),
                        "angular_velocity": (d["angular_velocity"] - truth["angular_velocity"]).abs().squeeze(-1),
                        "health": (d["health"] - truth["health"]).abs().squeeze(-1),
                        "power": (d["power"] - truth["power"]).abs().squeeze(-1),
                        "cooldown_s": (d["cooldown"] - truth["cooldown"]).abs().squeeze(-1),
                        "shield_delay_s": (d["shield_delay"] - truth["shield_delay"]).abs().squeeze(-1),
                        "local_log_index": (d["local_log_index"] - truth["local_log_index"]).abs().squeeze(-1),
                    }

                errs = {name: physical(d) for name, d in decoded.items()}
                # Dead reckoning and freezing, both from the last sighting:
                # elapsed is the age at the *next* state, one decision on.
                elapsed = (age + dt).unsqueeze(-1)
                true_pos = torch.cat([truth["position_x"], truth["position_y"]], dim=-1)
                for name, guess in (
                    ("dead_reckon", last_pos + last_vel * elapsed),
                    ("frozen", last_pos),
                ):
                    delta = torch.remainder(guess - true_pos + world / 2.0, world) - world / 2.0
                    errs[name] = {"position_px": delta.norm(dim=-1)}

                # per-harmonic resultant length of predicted / believed moments
                def resultant(targets, offset, n):
                    block = targets[..., offset : offset + 2 * n]
                    s = block[..., :n]
                    c = block[..., n:]
                    return torch.sqrt(s * s + c * c)

                res = {
                    "head_pos_x": resultant(pred_targets, pos_x.t_offset, n_harm),
                    "head_pos_y": resultant(pred_targets, pos_y.t_offset, n_harm),
                    "head_att": resultant(pred_targets, att.t_offset, n_att),
                    "belief_pos_x": resultant(curr_targets, pos_x.t_offset, n_harm),
                    "belief_att": resultant(curr_targets, att.t_offset, n_att),
                }

                not_alive = (~(alive_now & alive_next & ~respawned)).float()
                for cell, mask in cells.items():
                    acc.add(f"{cell}/tokens", torch.ones_like(age), mask)
                    acc.add(f"{cell}/dead_frac", not_alive, mask)
                    acc.add(f"{cell}/age_s", age, mask)
                    for model, e in errs.items():
                        for feat, value in e.items():
                            acc.add(f"{cell}/{model}/{feat}", value, mask)
                            acc.add(f"{cell}/{model}/sq_{feat}", value * value, mask)
                    # target-space per-dim squared error, scaled space
                    acc.add(f"{cell}/dim_sq/head", (pred_mean - labels).pow(2), mask)
                    acc.add(f"{cell}/dim_sq/identity", (id_pred_scaled - labels).pow(2), mask)
                    acc.add(f"{cell}/dim_sq/linear", (lin_pred_scaled - labels).pow(2), mask)
                    acc.add(f"{cell}/dim_sq/label", labels.pow(2), mask)
                    acc.add(f"{cell}/dim_var", variance, mask)
                    for key, value in res.items():
                        acc.add(f"{cell}/resultant/{key}", value, mask)

                    if cell.endswith("hidden"):
                        for lo, hi in zip(AGE_EDGES[:-1], AGE_EDGES[1:]):
                            bucket = mask & (age > lo) & (age <= hi)
                            tag = f"{cell}/age_{lo:g}_{hi:g}"
                            acc.add(f"{tag}/tokens", torch.ones_like(age), bucket)
                            for model in ("head", "identity", "linear", "dead_reckon", "frozen"):
                                for feat in ("position_px", "velocity_px_s", "attitude_rad"):
                                    if feat in errs[model]:
                                        acc.add(f"{tag}/{model}/{feat}", errs[model][feat], bucket)

                    # bounded samples of position error for percentiles
                    for model in ("head", "identity", "linear"):
                        key = f"{cell}/{model}/position_px"
                        bucket = samples.setdefault(key, [])
                        kept = sum(t.numel() for t in bucket)
                        if kept < sample_cap:
                            rows = errs[model]["position_px"][mask]
                            if rows.numel():
                                bucket.append(rows[: sample_cap - kept].to(torch.float16).cpu())

        runner.reset_finished(done_any)
        if (step + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  step {step + 1}/{NUM_STEPS}  {el:.0f}s  ({(step + 1) / el:.1f} steps/s)", flush=True)

    result = acc.result()
    pct = {}
    for key, chunks in samples.items():
        if not chunks:
            continue
        rows = torch.cat(chunks).float()
        pct[key] = {
            "n": int(rows.numel()),
            "p50": float(rows.median()),
            "p90": float(rows.quantile(0.9)),
            "p99": float(rows.quantile(0.99)),
        }
    payload = {
        "checkpoint": ckpt.name,
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "warmup": WARMUP,
        "decision_dt": dt,
        "feature_names": coord.get_feature_names(),
        "harmonics": {"position": n_harm, "attitude": n_att},
        "target_slices": {k: [v.start, v.stop] for k, v in coord.target_slices().items()},
        "label_scale": coord.label_scale_vector(torch.device("cpu")).tolist(),
        "metrics": result,
        "percentiles": pct,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
