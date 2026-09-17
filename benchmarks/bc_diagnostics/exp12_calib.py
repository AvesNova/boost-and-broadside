"""Exp 12: how accurate must the bearing be, and is there a left/right asymmetry?

Part 1 replaces Exp 6's Gaussian-marginalisation estimate with a fitted one. The
teacher's true bearing is corrupted by noise of a known scale, a turn head is
*fitted* on the corrupted bearing (so optimal hedging is learned rather than
assumed), and held-out turn KL is reported against the resulting error
distribution. That gives a directly usable requirement: to reach turn KL X, the
bearing must be known to Y degrees.

Part 2 asks whether the left/right gap in Exp 3 survives matching on |bearing|.
"""

import json
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp8_oracle import fit_head, kl  # noqa: E402
from exp11_intervention import latents, sincos  # noqa: E402


def noisy(t, sigma, gen):
    """Teacher bearings corrupted by a wrapped-Gaussian error of scale sigma (rad)."""
    out = {}
    for k in ("rel_front", "rel_personal"):
        out[k] = t[k] + torch.randn(t[k].shape, generator=gen) * sigma
    return out


def main():
    fit_tag, eval_tag = "heldout", "heldout2"
    L = {t: latents(t) for t in (fit_tag, eval_tag)}
    gen = torch.Generator().manual_seed(0)
    rows = []
    for sigma in (0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.2, 0.35):
        def feats(tag):
            t = L[tag]["teach"]
            n = noisy(t, sigma, gen)
            return torch.cat([
                sincos(n["rel_front"]), sincos(n["rel_personal"]),
                t["alpha"][:, None], n["rel_front"].abs()[:, None], n["rel_personal"].abs()[:, None],
            ], 1)

        lq = fit_head(feats(fit_tag), L[fit_tag]["expert"], feats(eval_tag))
        v = float(kl(L[eval_tag]["expert"], lq))
        # Report the error the same way the probes are reported, so the two
        # tables can be read against each other directly.
        e = (torch.randn(200000, generator=gen) * sigma).abs() * 180 / torch.pi
        rows.append(dict(
            sigma_rad=sigma, median_err_deg=round(float(e.median()), 2),
            p90_err_deg=round(float(torch.quantile(e, 0.9)), 2),
            heldout_turn_kl=round(v, 4),
        ))
        print(json.dumps(rows[-1]), flush=True)

    # --- left/right asymmetry, matched on |bearing| ---------------------------
    t = L[eval_tag]["teach"]
    exp, logq = L[eval_tag]["expert"], L[eval_tag]["logq"]
    pol_kl = (exp.clamp_min(1e-8) * (exp.clamp_min(1e-8).log() - logq)).sum(-1)
    eff = torch.where(t["alpha"] > 0.5, t["rel_front"], t["rel_combat"])
    a = eff.abs()
    asym = []
    edges = [0.0, 0.03, 0.06, 0.09, 0.12, 0.2, 0.3, 0.42, 0.7, 1.2, 2.0, 3.2]
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (a >= lo) & (a < hi)
        left, right = band & (eff < 0), band & (eff > 0)
        if int(left.sum()) < 100 or int(right.sum()) < 100:
            continue
        asym.append(dict(
            band=f"[{lo},{hi})", n_left=int(left.sum()), n_right=int(right.sum()),
            kl_left=round(float(pol_kl[left].mean()), 4),
            kl_right=round(float(pol_kl[right].mean()), 4),
            mean_abs_left=round(float(a[left].mean()), 4),
            mean_abs_right=round(float(a[right].mean()), 4),
        ))
        print(json.dumps(asym[-1]), flush=True)
    tot_l = float(pol_kl[eff < 0].mean())
    tot_r = float(pol_kl[eff > 0].mean())
    # Re-weight the right-hand bands to the left-hand |bearing| histogram, which
    # is what "matched on |bearing|" has to mean for an unequal-sized split.
    wl = torch.tensor([r["n_left"] for r in asym], dtype=torch.float)
    wl = wl / wl.sum()
    matched_r = float((wl * torch.tensor([r["kl_right"] for r in asym])).sum())
    matched_l = float((wl * torch.tensor([r["kl_left"] for r in asym])).sum())
    summary = dict(
        unmatched_kl_left=round(tot_l, 4), unmatched_kl_right=round(tot_r, 4),
        matched_kl_left=round(matched_l, 4), matched_kl_right=round(matched_r, 4),
        frac_left=round(float((eff < 0).float().mean()), 3),
    )
    print(json.dumps(summary), flush=True)
    json.dump(dict(calibration=rows, asymmetry=asym, asym_summary=summary),
              open(f"{S}/exp12_rows.json", "w"), indent=1)


main()
