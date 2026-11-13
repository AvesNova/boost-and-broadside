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
