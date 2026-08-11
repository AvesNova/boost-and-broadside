# Landmark checkpoint migration — `resilient-resonance-682`

One-time offline migration of the complete landmark checkpoint set into the schemas frozen by `S14R`. Produced by `scripts/migrate_682.py`; this file is the record the plan's phase 10 requires.

- Transformation version: `1`
- Training commit (recorded by the run in `wandb_export/files/wandb-metadata.json`): `b4883769ca49bb60e818986586db5673a4bf83c1`
- Observation schema written: `refractive_fields_v3`
- Files migrated: 16

## Derived run provenance

Every value below comes from the run's own recorded data or from the stored
tensors. Nothing is a placeholder, and `resolved_config` and `launch` are
omitted rather than invented — both are optional in the frozen schema and
neither was ever recorded for this run.

- `paradigm`: `ego_pass` — from the run's own `train_config["paradigm"]`.
- `model_config`: `{'d_model': 128, 'n_heads': 4, 'n_yemong_blocks': 2, 'n_spatial_per_block': 1, 'n_temporal_per_block': 1, 'encoder_split': False, 'n_bullet_cross_per_block': 0, 'bullet_encoder_hidden': 64, 'grad_checkpoint': False}`
- `env_config`: `{'num_ships': 8, 'max_bullets': 20, 'max_episode_steps': 1024, 'num_fields': 0, 'single_team': False, 'action_repeat': 1, 'spawn_resource_spread': 0.0}`
- `ship_config`: the training commit's `SHIP_CONFIG`, re-expressed in the current
  schema. Every field both versions define holds the same value, so the loader's
  physics-drift check correctly stays silent.

### Named unknowns

Values the current schema requires that the run never recorded. Each takes its
dataclass default, which is the absence of a choice rather than a measured one:

- `model_config.bullet_encoder_hidden` — inert while `n_bullet_cross_per_block=0`.
- `model_config.grad_checkpoint` — a backward-pass memory setting, not architecture.
- `ship_config.bullet_drag_coeff`, `ship_config.bullet_field_damage_scale`, `ship_config.bullet_field_integration_substeps`, `ship_config.bullet_field_integrator`, `ship_config.field_index_step`, `ship_config.field_integration_substeps`, `ship_config.field_integrator`, `ship_config.field_interface_damage`, `ship_config.field_radius_max`, `ship_config.field_radius_min`, `ship_config.field_transition_width_max`, `ship_config.field_transition_width_min` — refractive-field and bullet-drag physics, none of which existed at the training commit. The run has `num_fields=0`, and the four of these the feature pipeline reads scale only the zero-weighted encoder columns, so no value of theirs can reach these weights.

Fields the training commit defined that the current schema does not, and which
are therefore dropped rather than carried:

- `ship_config.bullet_collision_radius`
- `ship_config.obstacle_collision_radius`
- `ship_config.obstacle_gravity_harmonic`
- `ship_config.obstacle_radius_max`
- `ship_config.obstacle_radius_min`
- `env_config.num_obstacles` — the run set it to 0, so nothing is lost.

## Value-component mapping

The critic's eleven rows are the non-zero-weighted entries of the training
commit's `REWARD_COMPONENT_NAMES`, in that registry's order. Two of them are
spelled differently today, because the current environment splits projectile
damage from refractive-field boundary damage and the landmark run had no fields:

| K | historical name | current name |
|---:|---|---|
| 0 | `ally_win` | `ally_win` |
| 1 | `enemy_win` | `enemy_win` |
| 2 | `facing` | `facing` |
| 3 | `closing_speed` | `closing_speed` |
| 4 | `shoot_quality` | `shoot_quality` |
| 5 | `kill_shot` | `kill_shot` |
| 6 | `kill_assist` | `kill_assist` |
| 7 | `damage_taken` | `combat_damage_taken` |
| 8 | `damage_dealt_enemy` | `damage_dealt_enemy` |
| 9 | `damage_dealt_ally` | `damage_dealt_ally` |
| 10 | `death` | `combat_death` |

Ordering these eleven by the *current* registry's index reproduces the same
sequence, so **the value head needs no row permutation** and none is applied.
`scripts/migrate_682.py` recomputes this and raises rather than emitting a
scrambled critic if the two orders ever diverge.

## Changed input encodings

The one substantive discovery of this migration, and the only part of it that
is not a rename, a pad, or a copy. It is invisible in every shape, key, and
config field: the same physical quantity now reaches the encoder on a different
scale, through the same column.

| feature | column | divisor at training | divisor now | weight factor |
|---|---:|---:|---:|---:|
| `radius` | 57 | 40 | 512 | 12.8 |

`radius` was normalized by the obstacle radius ceiling (40) and is now normalized
by half the world size (512), because refractive fields are far larger than any
obstacle was. No `ShipConfig` field moved, so the loader's physics-drift check
cannot see this; it was found by diffing every shared feature's encoder
specification against the training commit, where it is the only difference.

Uncompensated, these weights would read a ship radius of 0.0195 where they were
fitted on 0.25. The compensation is exact, not approximate: the feature enters
through one column of one `Linear`, so scaling that column of the weight by
`512/40` reproduces the historical pre-activation for every input value. Adam's
moments for that column are carried as `1/k` and `1/k^2` (the gradient scales
inversely with the weight) and the averaging accumulator as `k`, so the
optimizer and averaged-policy records stay consistent with the weight.

## Per-file record

| file | family | original SHA-256 | migrated SHA-256 | migrated content SHA-256 |
|---|---|---|---|---|
| `step_000999424000.pt` | resumable | `c372dba9ce29ebbd3fc41b4f13ec4040037fb47dcf3aaa3942ae442b5896cb80` | `99d7338082ab25e6cc37d20a8c0453fa3fb81f1bc82da44f775aa25f85d62f70` | `9cc01b32008f077a3f94c762afdbb281703c912879e12cd2a53b81d17d085beb` |
| `recent_avg.pt` | resumable | `2831192c3b312e678ddebea8779f93b8eca557a0cabfbf9cb3451ccd666f986a` | `7f9c2462385005822dbeab301d1330e6785c76078d046785236344bfdf1095c1` | `f4825565c31aab1f55b02d59e06ef36e78d938cbc5f98d16951232d9c42987d1` |
| `best_training.pt` | best | `3c8c60bff631a6d0838c0a6ad8e5df41f0e13fccf37f9d5cf33c97d94169aded` | `e2975ba549eaf374609ea10200a325c6eb0c8e87c8cff64609e22137c523cc98` | `bf24128d3b50e62f68860136e9cefa7359f2ae9a0d44fd1c2985cbc64685faa1` |
| `ladder_step_000014991360.pt` | policy | `ef350d77c62845c3a6bd0d2b0620f3f8b3d6aa25ccac7afc6567cfcfb90d367f` | `c0b17022ec46995a5439ab30996583a86524b0699e61245740d2f788b523560f` | `feb8d6f9f02e4a04e3dbd05fac6183cb255a68dcabab49698a07adca816f4286` |
| `ladder_step_000021987328.pt` | policy | `636d218a0619cd9808ced2e4c4512dd454ba989d30dfbf6b6565a528a303f098` | `086805cdfa565bdeeb180af9f573ba8482066ad4f19121cc9ff1cc003a5acc45` | `f870e3d144c4c2729b68c0f4b982e85bcc3cc5a973ee36cd02e67bc90f95c602` |
| `ladder_step_000028983296.pt` | policy | `0d494ca99e5e910e1a2a5dfe480d8731162e773b594d311195ff60d7f2472360` | `c775625d484e6bae9de1c61f4f2ebb63def2bfd421242de2554e0f33b0365325` | `3879c6bb1839521db9659d7adbed0479c501d902238028790d96a104f173e4ef` |
| `ladder_step_000036978688.pt` | policy | `e9d3757d589d6033109e979f6a11125fcde4d400e19628dd684053d11e1705f6` | `75138ea7fddbd424eeee1f5bc65a68d2e0e8e539f16ef5104bb6f74f035c70a6` | `538853a15b5575bb6302f794fbed24e73c13981a998751d623ed2cfe0a4fda1e` |
| `ladder_step_000049971200.pt` | policy | `6879770afbf58db5bcd2ef94b6f4aa3cc4badf172d25bbf039324a586a0f8def` | `5cecf8bde7ae2429c5a77373a4dd46151736c0862664d6ea81a5482a0289dd0d` | `e8c3542a377540fac3c1ad8c45120f8582a5ad55052b9402f2cf2177d91ae557` |
| `ladder_step_000070959104.pt` | policy | `87716e57be5a09bd697d3990f3c5cb46915240636aabd8815fda922b9582c5ad` | `502d2d493fb29543edfd73a5a64ae9b8aecf3d8b351616109f1daeefc67bd6af` | `8f0094d069edeb77861a4bb17f81c761100c2078d8d755904a58212b84623376` |
| `ladder_step_000087949312.pt` | policy | `53cf4c3054bbde6d09f5d319544c6bd2de1ae78aeedb25c2d33fed9000b148f7` | `c96e707bede2a442174b68a3c3bdf36d8254da951a330eb003dc402dc3c12f95` | `37d257bbf67d977cbd01eb55befa154ee41cf0bd0827642b938a9f75249b0134` |
| `ladder_step_000102940672.pt` | policy | `dba0a383df4d899c95299308950250affc8eb8f76a8fe45aa1040969b1d0538b` | `57f00f5ee17d70908988eb4ac9dd91e115ed231ec2d9aad73d052a8d8f1c02d1` | `178c5bd5467b2983b8ddcd5a8c430ed9995cf712069e9a0cb777899078be7558` |
| `ladder_step_000155910144.pt` | policy | `4f24cfd6ab661ce99ea4792bf82be5caace68e92f6d0b15803fe12d71425d936` | `f2f4f75dc80d28ef2db3c195b2c4d534c9a4ff28fa0be2976ba19a5e52bda91a` | `a0c504a43d169098dd4028815af9a28082d7bc498a749b58e4988541dc75a9cb` |
| `ladder_step_000206880768.pt` | policy | `c661c34ae2ebf3471446aa313dcdfe91a543d70490af184ded1aa6aa754582a4` | `bdc084e7c7611b6e8bed9fab9c289c97038b15d5621c209657c492262fc4c273` | `c9e8cea11332c9fe8673e49a5c46d965e37a6c5598e4fd3b75ecb13545519f40` |
| `ladder_step_000272842752.pt` | policy | `d4e92a157631283251bf6187b6f60afd62200500f1069beb6e9bacb64103b934` | `ee1768b736c10a0ce476443693813093e2946feb59f803a04bb70bcd8959ba12` | `f3610eb7f4909fd54787d9a903353bf9e2320831153cac74c68575b024550729` |
| `ladder_step_000416759808.pt` | policy | `ae42a450d1f2c70afaa73115f3f1af533de99eebdf2fa89b64c8f9ddcbe277ff` | `fc7a75ffb51c37e8fbd083f906f34c745f5a81c09bd63fc8da6f9ef0c8999845` | `ce1aab66fc60a67f23a6e10d660cc6596b5a74b7515241e98c3b5042e22c990b` |
| `ladder_step_000876494848.pt` | policy | `e16c09edc16e4d96e258a19179f5ee020aff9712f7229e027ff0633d40a8f48e` | `0781b1ecdbf14cffab83e50cf4d836b428c12f1ff302557ac5ac21b96691f39f` | `4c4e2e3a62d6b10154c8d1096e5bf62a773736b377c28dad07c03fe58ff64572` |

The original hashes are the git-LFS object ids of the files this migration read;
`tests/migration/` re-derives them so the inputs stay identifiable. The migrated
hash identifies the bytes now tracked. The content hash is the one to compare when
*reproducing* the migration: the three payloads carrying the historical
`train_config` do not serialize byte-identically twice, because two of its reward
fields are `frozenset`s and `pickle` writes a frozenset in iteration order, which
Python randomizes per process. Their contents are identical; only the byte order of
two three-element sets moves. The other thirteen files are byte-reproducible.

### Transformations

Every file's `policy_state_dict` receives the same key rename, the same three
edits to existing tensors, and the same two introduced parameters; the two
resumable files additionally carry their optimizer and averaging state across it.

Key rename: `yemong_layers.<i>.spatial.<leaf>` -> `yemong_layers.<i>.spatial.0.<leaf>`
and the same for `temporal`. The block gained a list of sublayers and this run
trained with one of each.

`step_000999424000.pt`:

  - `encoder.feature_extractor.0.weight` [256, 58] -> `encoder.feature_extractor.0.weight` [256, 66]: scale column 57 by 12.8; pad input columns [58:66] with zeros
  - `next_state_head.net.3.weight` [9, 256] -> `next_state_head.net.3.weight` [10, 256]: pad prediction row 9 with zeros
  - `next_state_head.net.3.bias` [9] -> `next_state_head.net.3.bias` [10]: pad prediction row 9 with zeros
  - `(none)` None -> `yemong_layers.0.field_sub.0.weight` [128, 128]: introduced as identity (no legacy counterpart; inert with num_fields=0)
  - `(none)` None -> `yemong_layers.1.field_sub.0.weight` [128, 128]: introduced as identity (no legacy counterpart; inert with num_fields=0)

The list above is identical in every one of the sixteen files, so only the first
is shown here. The complete per-tensor mapping for every file — including the
69 tensors that are renamed or copied unchanged, and the per-parameter optimizer
index mapping for the two resumable files — is in `migration_report.json` beside
this file.

## Validation

Recorded per file in `migration_report.json` and enforced by
`tests/migration/test_landmark_682.py`, which runs over all sixteen files:

- the payload carries exactly its frozen family's key set;
- it loads through the ordinary `load_policy_bundle` with no migration path;
- the two resumable files pass `require_resumable_checkpoint`;
- fixed seeded observations through the migrated policy reproduce the historical
  policy's logits, action distributions, values, recurrent state, and next-state
  outputs, against a reference captured from the training commit;
- a seeded zero-field scenario reproduces the historical policy's actions and
  recurrent state over a multi-step episode.

Measured at transformation version 1, over all sixteen files and both input sets:
the encoder's own output agrees with the historical one to 2.4e-07, which is under
one float32 ULP at its magnitude. Amplified through two Yemong blocks that becomes
3.2e-05 on logits, 7.9e-06 on values, 8.6e-06 on the nine inherited next-state
outputs, and 1.0e-05 on the recurrent state. Every greedy action matches exactly.
Bitwise equality is not reachable and is not claimed: the encoder's first matmul
went from k=58 to k=66, which reorders the accumulation of the terms that survived.

## Known limitations

- **The tenth next-state predictor is zero.** The current head predicts a
  `local_log_index` target the historical head never had; its row is zero-padded,
  so these weights predict a constant zero for it. The other nine are exact.
- **`field_sub` has no history.** It is introduced as the identity every fresh
  block initializes it to. It is applied only to field tokens, and this run has
  `num_fields=0`, so it cannot affect any forward pass of these weights.
- **Resumable, with a caveat.** `step_000999424000.pt` and `recent_avg.pt` carry
  their complete Adam state and its recorded hyperparameters (`lr=1e-4`,
  `eps=1e-5`) across the rename, so they satisfy the frozen resumable contract.
  Actually continuing the run additionally needs a profile whose reward vocabulary
  matches the historical one, which no longer exists — the current registry splits
  two of these eleven components. These files are complete and loadable; they are
  not a resumable path back into today's training system.
- **`train_config` is kept verbatim**, in the historical schema. It is the record
  of what the run was launched with. No loader rebuilds it into a dataclass.
