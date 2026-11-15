import collections, json, statistics, sys
games = collections.defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line); games[r["label"]] += r["games"]
def q(xs, p): xs = sorted(xs); return xs[min(int(p * len(xs)), len(xs) - 1)]
print("radius |  mean median  p80  p90 | never ended | deaths/min (both) | boundary share")
for label, g in sorted(games.items(), key=lambda kv: -int(kv[0][1:])):
    t = [x["end_tick"] / 30 if x["by_rule"] else 892 for x in g]   # unfinished: collapse at clock - 8 s
    ended = [x for x in g if x["by_rule"]]
    deaths = sum(x["deaths"] for x in ended); bd = sum(x["boundary_deaths"] for x in ended)
    minutes = sum(x["end_tick"] / 1800 for x in ended)
    print(f"{label[1:]:>6} | {statistics.mean(t):5.0f} {statistics.median(t):6.0f} {q(t,.8):4.0f} {q(t,.9):4.0f} |"
          f" {1 - len(ended)/len(g):11.2f} | {deaths/max(minutes,1e-9):17.1f} | {bd/max(deaths,1):14.3f}")
