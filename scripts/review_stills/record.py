"""Record one Imugi-vs-Imugi match to a capture file (scratch: review stills)."""
import sys, time
import torch
from bnb.figures import match as matches, capture

preset, ships, map_rule, seed, seconds, out = sys.argv[1:7]
torch.set_num_threads(int(sys.argv[7]) if len(sys.argv) > 7 else 14)
config = matches.MatchConfig(
    preset=preset, ships_per_team=int(ships), map_rule=map_rule, seed=int(seed),
    seconds=None if seconds == "none" else float(seconds),
)
started = time.time()
artifact = matches.record(matches.sides_of(None, config), config)
m = artifact["match"]
print(f"{out}: {len(m['clock']) * m['about']['tick_length']:.0f} s game in {time.time() - started:.0f} s;"
      f" deaths {int(m['died'].sum())}", flush=True)
capture.save(artifact, out)
