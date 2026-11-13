# Yongma and Jujak under r9m2: progress

Branch `tune/ships-r9m2` (from `origin/eval/r9m2-maps` 22764d67, with `bench/game_loop.py`'s
type-row lineups from `origin/eval/morning`). The task: re-balance the original two ships,
Yongma (energy fighter) and Jujak (turn fighter), under the scripted doctrine r9m2, keeping
Jujak the better turner (sustained and instantaneous) and Yongma the faster, tougher,
harder-hitting, better-diving ship; and a three-ship set with a middle ship. Hard stop 13:00
KST. GPU: gpubox7 (RTX 5060 Ti 16 GB, shared with a ratings job).

## Method

- `make.py <round>`: cards as factors on the frozen ship files (the Yongma slot on
  `yongma.toml`, the Jujak slot on `jujak.toml`), flown by r9m2 (`maneuvers-2/champion2.json`)
  with the Jujak slot's duel-extension gate at 0.7 (`g0.7`) or 1.0 (`g1.0`).
- maneuvers-2's `sizes.py`, unchanged: self-play, 300 s games, fog on, the density rule's arena;
  512 games a cell while searching (10v10 256, 20v20 128), 1024 to confirm. Lineups: energy v
  turn (Yongma's score, win = more kills, draws half) at each size, both 1v1 mirrors.
- `flight.py`: each card's sustained turn (full stick, settled), instantaneous peak turn,
  cruise, top and corner speed.
- `threeship.py`: the three-ship set in one card table (type-row lineups).
- `remote.sh`: a run in tmux on gpubox7.

Kill mix m / o / f / t: mutual head-on, one-sided head-on, flank, tail.

## 08:44-09:10 KST: round 0, the baseline (`round0/`)

512 games (10v10 256, 20v20 128). Yongma's score 1v1 / 2v2 / 5v5 / 10v10 / 20v20; mirrors at
1v1: mutual head-on share, life, post-merge choice entropy H (bits).

| Card, Jujak gate | Y v J | YY mirror | JJ mirror | 5v5 life, kills/min/ship |
|---|---|---|---|---|
| frozen, 0.7 | 0.01 / 0.45 / 0.29 / 0.17 / 0.05 | 0.60, 281 s, 1.59 | 0.70, 136 s, 1.26 | 156 s, 0.36 |
| frozen, 1.0 | 0.01 / 0.42 / 0.30 / 0.12 / 0.03 | 0.58, 272 s, 1.58 | 0.80, 144 s, 1.60 | 161 s, 0.35 |
| d2, 0.7 | 0.48 / 0.46 / 0.59 / 0.53 / 0.21 | 0.60, 186 s, 1.60 | 0.53, 230 s, 0.77 | 131 s, 0.43 |
| d2, 1.0 | 0.50 / 0.46 / 0.53 / 0.53 / 0.20 | 0.58, 192 s, 1.60 | 0.46, 225 s, 1.58 | 133 s, 0.42 |

The frozen card under r9m2 is Jujak's at every size but 2v2 (r9: 0.01 at 1v1 as well): Jujak
wins every duel (kills on Yongma's flank and tail, 0.42 / 0.29) and the crowds, and its mirror
is a quick joust (0.70-0.80 mutual, 136-144 s). The gate does not move the balance.

Flight (`flight.py`): frozen Yongma 0.241 rad/s sustained, 0.568 peak, cruise 167 m/s, corner
144 m/s; Jujak 0.311 / 0.661, 158 m/s, 125 m/s. Jujak's induced drag lowers only its
sustained turn (x1.3: 0.266, +10% over Yongma; x1.4: 0.254, +5%; x1.6 (d2): 0.232, Yongma
the better turner) and its cruise (x1.4: 144 m/s).

## 08:46-09:45 KST: screens (rounds 1-6, 512 games, gate 0.7)

What moves what, Yongma's score (1v1 / 5v5 / 10v10 unless named):

1. **Jujak's induced drag is a crowd lever more than a duel lever**: x1.3 0.12 / 0.67 / 0.66,
   x1.4 0.24 / 0.79 / 0.72 (frozen 0.01 / 0.29 / 0.17).
2. **Yongma damage and shield** add a little to the duel (+0.05-0.09 a x0.1-0.2) and a lot to
   the crowds (Jd1.4 Ydmg1.15 Ysh1.15: 0.32 / 0.86 / 0.89). Range x1.15 does nothing.
3. **Yongma's nose cone (f0) hardly moves the duel** (Jd1.4, f0 x0.6 / 0.4 / 0.25: 0.24 /
   0.24 / 0.27).
4. **Yongma's lift (its instantaneous turn only; sustained unchanged) is the duel lever**:
   x1.2 (peak 0.623 rad/s, Jujak's 0.661 still 6% higher) takes Jd1.4 Ydmg1.3 from 0.31 to
   0.48 at 1v1, and costs crowds a little (0.83 → 0.81 at 5v5).
5. **Yongma's dive exchange is the crowd counter, steeper with size**: Jd1.4 Yl1.2 Ydmg1.3 with
   dive x1.3 / 1.4 / 1.5: 0.51 / 0.45 / 0.44 at 1v1, 0.64 / 0.58 / 0.55 at 5v5, 0.53 / 0.45 /
   0.27 at 10v10, 0.20 / 0.12 / 0.09 at 20v20. **It also raises Yongma's settled sustained
   turn** (x1.5: 0.248 rad/s), so with Jujak drag x1.4 Jujak is only +2.4%: the cards from
   round 5 on use drag x1.35 (+5-6%) or x1.3 (+7-8%).
6. **A softer Jujak nose (f0 x1.5-2.0) is a duel lever with a small crowd cost**: Jd1.35 Yl1.2
   Ydmg1.3 Ydive1.5 with Jujak f0 x1.5 / x2.0: 0.56 / 0.63 at 1v1, 0.59 / 0.69 at 5v5, 0.33 /
   0.49 at 10v10.
7. **2v2 is Yongma's best size on every such card** (round 6: 0.66-0.74 with 1v1 0.48-0.58,
   5v5 0.53-0.67, 10v10 0.34-0.53), and **20v20 Jujak's** (0.07-0.20).

## 09:42-10:00 KST: round 7 (single levers) and the map size (`round7/`, `arena1/`, `arena_screen/`)

Round 7, single levers on Jd1.35 Yl1.2 Ydmg1.1 Ydive1.2 Jf1.5 (0.59 / 0.78 / 0.65 / 0.51 at
1v1 / 2v2 / 5v5 / 10v10): none takes 2v2 down alone. Yongma sight x0.9 is the strongest
(0.42 / 0.69 / 0.57 / 0.43), Jujak shield x1.15 next (0.46 / 0.66 / 0.60 / 0.42); Jujak sight
x1.1, Yongma range x0.85, Yongma hit radius x1.1, Jujak damage x1.1 and Yongma power x0.9
move every size by 0.02-0.11 the same way.

**The map size is the dominant axis** (the author's request: 5v5 at R 3.9 / 7.8 / 15.6 km, 1v1
at R 3.9 / 15.6 km; `arenas.sh`, `arena5.sh`). Round 6's Jd1.35 Yl1.2 Ydmg1.1 Jf1.5 with dive
x1.3 / x1.2:

| Card | 5v5 R 3.9 / 7.8 / 15.6 km | 1v1 R 3.5 (density rule) / 3.9 / 15.6 km |
|---|---|---|
| Ydive1.3 | 0.97 / 0.57 / 0.12 | 0.54 / 0.45 / 0.42 |
| Ydive1.2 | 0.97 / 0.67 / 0.15 | 0.59 / 0.46 / 0.42 |

The fleet sizes follow the same axis through the density rule's radius (1v1 3.5 km, 2v2
4.9 km, 10v10 11 km, 20v20 15.6 km): 2v2 is Yongma's best size and 20v20 Jujak's because the
map is small and large. The 1v1 at 15.6 km is mostly draws (the Yongma mirror scores no kill
in 300 s).

Arena screen, single levers on the Ydive1.2 card at 5v5 R 15.6 km (R 3.9 km saturates at
0.97-1.00 for every one of them): base 0.15; Yongma dive x1.0 (back to the frozen exchange)
0.36, engine power x1.15 0.36, sight x1.2 0.24, climb x1.3 0.24, parasitic drag x0.85 (cruise
179 m/s) 0.23, Jujak parasitic drag x1.15 0.22, Jujak sight x0.85 0.21, Yongma recharge x2
0.13, Jujak recharge x0.5 0.15. Speed, not dive, is what Yongma needs on the large map.

## 10:00-10:20 KST: speed for the large map (rounds 8-9), the reference cards across the map

Round 8 (`round8/`, `round.sh`): Yongma engine power in place of the dive exchange, damage and
shield down. Round 9 (`round9/`, `radius.sh`): bigger trades. Frozen and D2 across the map
(`radius0/`). 512 games; 5v5 at R 3.9 / 7.8 / 15.6 km, and (round 8) 1v1 and 10v10 on the
density rule. All Jd1.35 Yl1.2 Jf1.5 unless named.

| Card | 5v5 R 3.9 / 7.8 / 15.6 km | 1v1 | 10v10 |
|---|---|---|---|
| frozen | 0.70 / 0.29 / 0.05 | 0.01 | 0.17 |
| d2 | 0.84 / 0.59 / 0.10 | 0.48 | 0.53 |
| Ydmg1.1 Ydive1.3 (round 6) | 0.97 / 0.57 / 0.12 | 0.54 | 0.45 |
| Ydmg1.1 Ypow1.15 | 0.99 / 0.88 / 0.55 | 0.64 | 0.90 |
| Ydmg0.9 Ypow1.15 | 0.96 / 0.82 / 0.46 | 0.51 | 0.79 |
| Ydmg1.1 Ysh0.8 Ypow1.15 | 0.99 / 0.80 / 0.43 | 0.62 | 0.81 |
| Ydmg0.9 Ysh0.8 Ypow1.15 | 0.96 / 0.67 / 0.32 | 0.46 | 0.65 |
| Ydmg0.9 Ysh0.8 Ypow1.3 | 0.97 / 0.79 / 0.52 | 0.50 | 0.87 |
| Ydmg0.9 Ysh0.8 Ypow1.15 Jf2.0 | 0.98 / 0.77 / 0.36 | 0.68 | 0.77 |
| Ydmg0.9 Ysh0.8 Ypow1.3 Ydive1.2 | 0.96 / 0.63 / 0.37 | 0.48 | 0.64 |
| Ysh0.6 Ypow1.4 | 0.98 / 0.83 / 0.51 | | |
| Ysh0.6 Ydmg0.8 Ypow1.5 Ydrag0.8 | 0.94 / 0.77 / 0.61 | | |
| Jsh1.5 Ypow1.3 | 0.92 / 0.72 / 0.50 | | |
| Jsh1.5 Jdmg1.2 Ypow1.4 Yvis1.2 | 0.98 / 0.78 / 0.62 | | |
| Yrad1.3 Ypow1.3 | 0.99 / 0.87 / 0.68 | | |
| Ysh0.5 Ydmg1.2 Yrange0.75 Ypow1.4 | 0.97 / 0.79 / 0.40 | | |
| Ysh0.6 Ypow1.4 Yvis1.2 Jvis0.85 | 1.00 / 0.94 / 0.73 | | |
| Ysh0.7 Ydrag0.75 (power x1.0) | 0.92 / 0.62 / 0.22 | | |

- **The small-to-large lean is structural**: every card, the frozen one and D2 included, is
  0.65-0.90 more Yongma's at R 3.9 km than at 15.6 km. Speed (power, cruise) narrows it from
  0.85 (round 6) to 0.33-0.45; nothing tried moves the small map below 0.92 without also
  taking the standard map below 0.6.
- **Speed is Yongma's large-map lever, the dive exchange works against it** (and against
  10v10-20v20); damage, shield and Jujak's toughness move all three radii together.
- Yongma's hit radius x1.3 (a bigger target) *helps* Yongma (0.68 on the large map): a
  bigger hit radius is not a nerf under this doctrine (not looked into).

## 09:59-10:07 KST: the three-ship set (`threeship/`)

`threeship.py` (type-row lineups): Yongma, Jujak and the middle ship M, the geometric mean of
the card's Yongma and Jujak in flight, shield, damage, size and nose cone (`make.middle_of`).
Each M flies either the Yongma doctrine (Me) or the Jujak one (Mt). 512 games (10v10 256),
r9m2 with Jujak gate 0.7. Side A's score (the first named ship's):

| Lineup | Frozen card: 1v1 / 5v5 / 10v10 | Card B (Jd1.35 Yl1.2 Ydmg1.1 Ydive1.2 Jf1.5): 1v1 / 5v5 / 10v10 |
|---|---|---|
| Y v J | 0.01 / 0.31 / 0.18 | 0.56 / 0.67 / 0.54 |
| Y v Me | 0.12 / 0.57 / 0.47 | 0.47 / 0.44 / 0.35 |
| J v Me | 0.97 / 0.55 / 0.56 | 0.35 / 0.20 / 0.19 |
| Y v Mt | 0.08 / 0.35 / 0.24 | 0.45 / 0.52 / 0.36 |
| J v Mt | 0.98 / 0.51 / 0.61 | 0.36 / 0.23 / 0.23 |
| M mirror (Me; Mt) 1v1 mutual, life, H | 0.71, 189 s, 1.59; 0.61, 168 s, 1.25 | 0.77, 153 s, 1.59; 0.65, 144 s, 1.24 |

Mixed lineup (2 Y + 2 J + 1 M a side; symmetric, so A ≈ 0.50), per-type K/D on card B: 5v5 Y
0.84, J 1.10, M 1.08 (Me) / 1.11 (Mt); 10v10 Y 0.75, J 1.19, M 1.06-1.07.

- **The geometric-mean middle ship is too strong on card B**: it beats Jujak at every size
  (0.19-0.36) and Yongma in the crowds (0.35-0.47). M is the best sustained turner of the
  three (0.251 rad/s, against Jujak's 0.260 it is close, against Yongma's 0.245 ahead) with
  the middle shield and damage. On the frozen card (Jujak 0.311 sustained) Jujak beats it
  0.97 in the duel. A nerf (shield or damage x0.85-0.9) was not run.
- Which doctrine M flies matters less than the card (Me and Mt within 0.08 of each other).

## State (10:25 KST): stopped for the flight-model change

The author is changing the flight model (no store charging while turning, a capped boost,
sustained turns at 70-95 m/s), which changes every card; the 1024-game confirmation was not
run. Everything here is 512 games, r9m2 with Jujak gate 0.7 (1.0 changes nothing in the
balance; it only changes the Jujak mirror). SE about ±0.015 at 1v1, ±0.02 at 5v5, ±0.03 at
10v10, ±0.035 at 20v20; the run-to-run spread is about twice that.

**What moved what** (for the re-balance on the new flight model):

1. **The map radius dominates the fleet size.** Small maps are Yongma's and large maps are
   Jujak's on every card (frozen 0.70 → 0.05, D2 0.84 → 0.10, 5v5 at R 3.9 → 15.6 km). The
   fleet-size pattern (2v2 Yongma's best, 20v20 Jujak's) is the density rule's radius (1v1
   3.5 km, 2v2 4.9 km, 10v10 11 km, 20v20 15.6 km). The 1v1 is the exception: it barely
   moves with the radius (0.54 → 0.45 → 0.42, the large-map duel mostly draws).
2. **The duel (1v1)**: sustained turn decides it; Jujak's +5-8% was worth 0.1-0.25 to
   Jujak here. Yongma's lift (instantaneous turn, x1.2, still 6% below Jujak's) and a
   softer Jujak nose (f0 x1.5-2.0) are the duel levers that cost the crowds least; damage
   helps a little, the Yongma nose cone, range and dive hardly at all.
3. **The crowds and the large map**: Jujak's induced drag, Yongma damage and shield move
   every size; Yongma speed (engine power, cruise) moves the large map most; Yongma's dive
   exchange takes the crowds and the large map away from Yongma (steeper with size) and
   raises its sustained turn (a dive x1.5 cost Jujak half its turn edge).
4. **The small map** (5v5 at 3.9 km) stays 0.92-1.00 Yongma on every card with Jujak still
   the better turner; the least-bad levers are Jujak shield x1.5 and Yongma shield x0.6-0.7.
5. **Mirrors**: a softer Jujak nose makes the Jujak mirror a first-pass joust (f0 x1.5: 0.74
   mutual, 103 s, 0.68 of kills on the first pass, H 1.36; frozen 0.70, 136 s); the Yongma
   mirror is 0.50-0.65 mutual, 200-300 s, H 1.6.
6. **The middle ship** at the geometric mean is too strong in crowds once Jujak's turn edge
   is small; it needs a toughness or damage nerf.

**Interim cards as maps** (`bnb play --preset <name>`; lineup 2 Yongma + 3 Jujak; factors on
the frozen files; the ship files untouched):

| Map | Card | Fleet 1v1 / 2v2 / 5v5 / 10v10 / 20v20 | 5v5 R 3.9 / 7.8 / 15.6 km | Yongma: sustained, peak, cruise, corner | Jujak |
|---|---|---|---|---|---|
| `yj-dive` | Jd1.35, Y lift x1.2, dmg x1.1, dive x1.3; J nose f0 x1.5 | 0.54 / 0.73 / 0.57 / 0.45 / 0.13 | 0.97 / 0.57 / 0.12 | 0.246, 0.623 rad/s, 167, 132 m/s | 0.260, 0.661, 146, 125 |
| `yj-speed` | Jd1.35, Y lift x1.2, dmg x0.9, shield x0.8, power x1.15; J f0 x1.5 | 0.46 / - / 0.67 / 0.65 / - | 0.96 / 0.67 / 0.32 | 0.246, 0.625, 167, 132 | same |
| `yj-flat` | Jd1.35, Y lift x1.2, dmg x0.8, shield x0.6, power x1.5, parasitic drag x0.8; J f0 x1.5 | not run | 0.94 / 0.77 / 0.61 | 0.250, 0.628, 184, 132 | same |

`yj-dive` is the fleet-size candidate (1v1-10v10 within 0.45-0.57 but 2v2 0.73), `yj-speed`
the large-map compromise, `yj-flat` the flattest across the map radius (spread 0.33) but
Yongma-leaning and barely tougher than Jujak (shield 85 against 80, damage equal). None is
balanced across the map radius.
