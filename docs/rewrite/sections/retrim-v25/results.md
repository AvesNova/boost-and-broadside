# Re-trim for the committed-turn Imugi (game v25)

The committed turn (feat/imugi-initiative) moved every team mode toward Yongma by 0.025-0.05 of
the kill share on game v24's card (Y0.46, head-on 0.05). Three trims of v24's card
(retrim.py, trims.json), both Imugi rules on, `bench/action_mix.py`, 512 games each, eager;
Yongma's kill share / point share (Y0.46's line is imugi-initiative's "both" column; SE of the
kill share about 0.005 in DM 2v2, 0.003 in DM 5v5, 0.014 in Elim 5v5, 0.004 in FL 5v5).

| card | DM 2v2 (.481-.521) | DM 5v5 (.494-.507) | Elim 5v5 (.498-.545) | FL 5v5 (.491-.513) | in band | worst miss |
|---|---|---|---|---|---|---|
| Y0.46 (v24) | 0.466 / 0.455 | 0.514 / 0.513 | 0.592 / 0.579 | 0.507 / 0.508 | 1 | 0.047 |
| Y0.42 | 0.458 / 0.445 | 0.500 / 0.500 | 0.584 / 0.564 | 0.499 / 0.499 | 2 | 0.039 |
| Y0.38 | 0.438 / 0.430 | 0.497 / 0.495 | 0.535 / 0.521 | 0.487 / 0.487 | 2 | 0.043 |
| Y0.42-J1.22 | 0.449 / 0.436 | 0.499 / 0.497 | 0.556 / 0.543 | 0.497 / 0.498 | 2 | 0.032 |

Y0.42 = Yongma's max_shield factor 0.42 (of v23's; v24 had 0.46), Y0.38 = 0.38, J1.22 = Jujak's
1.22 (v24: 1.1586).

## Choice: Y0.42-J1.22 (game v25)

Every trim has two conditions in band; Y0.42-J1.22 has the smallest worst miss (DM 2v2, 0.032
under). Its Elim 5v5 is 0.011 over its band (under one SE); Y0.38 is the only card in band there,
but at the cost of DM 2v2 (0.043 under) and FL 5v5 (0.004 under). DM 2v2 stays Jujak's on every
card: a shield trim does not fix it, and it moves the wrong way as Yongma's shield falls.
