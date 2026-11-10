# The GPU box's runs on the Hugging Face Hub

The RTX 3090 box (Vast.ai, 2026-11-08 to 2026-11-10) kept its large files on the Hub and its
small records here. The box itself is gone; everything it made is in one of the two places.

**Hub:** <https://huggingface.co/julianzia/boost-and-broadside-runs> (public model repo),
checked at revision `c07157076b36ecb71dbb9b1b9e4a257bfb1ca9eb` on 2026-11-10 23:05 KST.

| Hub path | What |
|---|---|
| `runs/<run>/` | each run directory whole: `checkpoints/` (training states), `policies/step_*/{live,uniform_average,exponential_average}/` (policy files), `policy-*/` (a BC run's final policies), `league/`, `run.json`, `config.json`, `metrics.jsonl` |
| `ratings/<tournament>/` | the tournaments' batch records and leaderboards |

**Here (git, `results/gpu-box/`):**

| Path | What |
|---|---|
| `hub-manifest.jsonl` | every file on the Hub at that revision: path, size, SHA-256 (LFS files) or git blob id (others). Line 1: the repo, the revision and the file count. |
| `runs/<run>/` | each run's `run.json`, `config.json`, `metrics.jsonl` and console (`console.txt`) |
| `ratings/` | the tournaments' records and console logs (`ratings-*.txt`) |
| `crossover/` | crossover fights and win rates at fixed odds (`fixed-odds/*.json`, `adaptive/`, `summary.jsonl`, per-run `artifacts/crossover/*/result.json`) |
| `logs/` | every runner's console (`*.out.txt`, `*.nohup.txt`, warnings stripped), the scorecards |
| `failed/` | aborted runs' records |
| `evidence/` | small records the log cites from the box's scratch space: the 40v40 smoke test, the crossover GPU check and the illegal-memory-access investigation (`xo-null/`), the champion's fresh-cache re-check, profiling for the SPS work (`prof-*.txt`, `t-*.log`) |
| `runner.txt`, `scripts/` | the runner log (UTC times) and the runner scripts |
| `*.png` | the figures |

The full account of what was done, why and when (KST): `docs/rewrite/sections/gpu-box-log.md`;
the tools: `docs/rewrite/sections/gpu-box/`.

**Integrity.** Before the box was returned, `gpu-box/hub_verify.py` compared every local file the
Hub should hold (4143) with the Hub's copy, byte for byte (SHA-256 for LFS files, the git blob id
for others): none missing, none different. Policy and checkpoint files also carry their own
SHA-256 in their `checkpoint.json`; `gpu-box/hub_repair.py` checks the Hub against those. Files
deleted on the box during the runs (per-update RL policy snapshots, older checkpoints) were
deleted only after their Hub copies matched those records; they are on the Hub only.

**Fetching:** `huggingface_hub.snapshot_download("julianzia/boost-and-broadside-runs",
revision="c07157076b36ecb71dbb9b1b9e4a257bfb1ca9eb", allow_patterns=["runs/<run>/policies/step_<n>/uniform_average/*"])`,
or `docs/rewrite/sections/gpu-box/hub_fetch.py`.
