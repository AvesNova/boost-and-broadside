---
covers: []
---

# References

The published work Boost and Broadside builds on or uses to make decisions, grouped by topic. The BibTeX is in [`references.bib`{.plain}](references.bib), and each entry here gives its key. Each line says what the project uses the work for and where it matters in the repository. "v1" means the first version's docs on `main`.

Every entry was checked on 2026-11-16 against a primary record: arXiv, the publisher's DOI record, the venue's proceedings, or NASA's technical reports server. The last section lists what could not be checked that way. `docs/rewrite/references.md` is the longer working list, which records how each work affected a decision.

## Reinforcement learning objective

- **PPO**, Schulman et al. 2017 (`schulman2017proximal`{.plain}): the clipped surrogate objective, the base RL loss. `src/bnb/learn/losses.py`, `src/bnb/learn/ppo.py`, `explanation/learner.md`; v1 `docs/training.md`{.plain}.
- **GAE**, Schulman et al. 2016, ICLR (`schulman2016high`{.plain}): lambda-weighted advantages, with one discount and one lambda per reward level. `src/bnb/learn/returns.py` (`per_level_gae`), `explanation/learner.md`.
- **DAPO**, Yu et al. 2025, NeurIPS (`yu2025dapo`{.plain}): clip-higher, an asymmetric ratio clip (`trust_region = "clip_higher"`). `src/bnb/learn/ppo.py`, `explanation/learner.md`.
- **DPPO**, Qi et al. 2026, ICML (`qi2026rethinking`{.plain}): a mask on a direct estimate of the divergence between policies, in place of ratio clipping (`dppo_tv`, `dppo_binary_tv`). `src/bnb/learn/losses.py` (`dppo_keep`), `explanation/learner.md`, `rewrite/sections/dppo-notes.md`.
- **Time limits**, Pardo et al. 2018, ICML (`pardo2018time`{.plain}): the reason the observed game clock is a real termination, with no bootstrap through it. `rewrite/yemong-review.md`.
- **Potential-based shaping**, Ng, Harada and Russell 1999, ICML (`ng1999policy`{.plain}): v1's shaping terms. v1 `docs/training.md`{.plain}.
- **R2D2 burn-in**, Kapturowski et al. 2019, ICLR (`kapturowski2019recurrent`{.plain}): stale recurrent carries and burn-in, weighed for the learner's sequences. `rewrite/yemong-review.md`.

## Behaviour cloning, league and self-play

- **DAgger**, Ross, Gordon and Bagnell 2011, AISTATS (`ross2011reduction`{.plain}): the teacher labels the states the student visits. Cloning is on-policy in this sense, though without mixing in the teacher's actions. `explanation/behaviour-cloning.md`, `src/bnb/runtime/cloning.py`.
- **AlphaStar**, Vinyals et al. 2019, Nature (`vinyals2019grandmaster`{.plain}): supervised policy first, then RL fine-tuning against a league; prioritised fictitious self-play weighting; a KL term toward the supervised policy. `src/bnb/league/pool.py`, `rewrite/decisions.md`, `rewrite/yemong-review.md`.
- **OpenAI Five**, Berner et al. 2019 (`berner2019dota`{.plain}): training against past selves, and team-spirit mixing of advantages (kept as config, set to 0). `src/bnb/learn/returns.py` (`team_spirit`), `src/bnb/league/`.
- **AlphaZero**, Silver et al. 2018, Science (`silver2018general`{.plain}): one network trained on both players' data in self-play. `src/bnb/learn/ppo.py`, `rewrite/arms-todo.md`.
- **SMAC**, Samvelyan et al. 2019, AAMAS (`samvelyan2019starcraft`{.plain}): the benchmark behind MAPPO's results, cited in the privileged-critic discussion. `rewrite/yemong-review.md`.
- **MAPPO**, Yu et al. 2022, NeurIPS Datasets and Benchmarks (`yu2022surprising`{.plain}): PPO with weights shared across agents, as Yemong shares them across ships; also a reference for centralised critics. `rewrite/yemong-review.md`.

## Critics under partial observability

These are discussion only: the values use the actor's own fogged view.

- **Asymmetric actor-critic**, Pinto et al. 2018, RSS (`pinto2018asymmetric`{.plain}): a critic with privileged state. `rewrite/yemong-review.md`.
- **Lyu et al. 2021, AAMAS** (`lyu2021contrasting`{.plain}), **2022, AAAI** (`lyu2022deeper`{.plain}), **2023, JAIR** (`lyu2023centralized`{.plain}): centralised critics are not uniformly better (bias against variance). `rewrite/yemong-review.md`.
- **Baisero and Amato 2022, AAMAS** (`baisero2022unbiased`{.plain}): a critic of the state alone biases the policy gradient in a POMDP; the unbiased form takes history and state. `rewrite/yemong-review.md`.

## Weight averaging

- **SWA**, Izmailov et al. 2018, UAI (`izmailov2018averaging`{.plain}): the windowed uniform tail average in the averaging sweep. The league's uniform and exponential averages are close relatives. `rewrite/sections/averaging/suite.toml`, `src/bnb/league/pool.py`, `explanation/league.md`.
- **LAWA**, Kaddour 2022, NeurIPS HITY workshop (`kaddour2022stop`{.plain}): averaging the latest k checkpoints, the planned averaging sweep. `rewrite/arms-todo.md`.

## Ratings and evaluation

- **Bradley–Terry**, Bradley and Terry 1952, Biometrika (`bradley1952rank`{.plain}): every rating is a Bradley–Terry maximum-likelihood fit, with the scripted Imugi pinned at 1000. `src/bnb/league/ratings.py`, `design.py`, `explanation/league.md`; v1 `docs/evaluation.md`{.plain}.
- **Elo**, Elo 1978 (`elo1978rating`{.plain}): the 400-point logistic scale used for reporting. There is no K-factor update. `explanation/league.md`.
- **Thurstone**, 1927, Psychological Review (`thurstone1927law`{.plain}): the probit link, one of the models compared in the goodness-of-fit checks. `src/bnb/league/cyclic.py`, `explanation/game/real-scale-balance.md`.
- **mElo**, Balduzzi et al. 2018, NeurIPS (`balduzzi2018reevaluating`{.plain}): multidimensional Elo (mElo-2k), used to find cyclic structure in balance work and for the loop check against Bradley–Terry. `src/bnb/league/cyclic.py`, `goodness_of_fit.py`, `explanation/game/real-scale-balance.md`, `explanation/game/ship-design.md`.
- **Optimal design**: Kiefer and Wolfowitz 1960 (`kiefer1960equivalence`{.plain}), the equivalence theorem the A-optimal design test checks. Silvey, Titterington and Torsney 1978 (`silvey1978algorithm`{.plain}), the multiplicative algorithm. The code names the method but cites no paper, so this is the suggested citation. `src/bnb/league/design.py`, `tests/sections/test_rating_design.py`, `explanation/league.md`.
- **Learning-curve extrapolation**: Domhan, Springenberg and Hutter 2015, IJCAI (`domhan2015speeding`{.plain}) and Jones 2021 (`jones2021scaling`{.plain}). Considered for judging arms early, then rejected in favour of the endpoint Elo. `rewrite/references.md`, `rewrite/arms-todo.md`.

## Sequential testing (arms decisions)

`bnb evaluate` uses a group-sequential test with O'Brien–Fleming-type error spending (Lan–DeMets), not Pocock boundaries.

- **Armitage, McPherson and Rowe 1969**, JRSS A (`armitage1969repeated`{.plain}): recursive numerical integration of the boundary-crossing probabilities. `src/bnb/league/sequential.py`.
- **O'Brien and Fleming 1979**, Biometrics (`obrien1979multiple`{.plain}): the shape of the alpha-spending function. `rewrite/decision-rule.md`, `rewrite/evaluation-suite.md`, `src/bnb/league/sequential.py`.
- **Lan and DeMets 1983**, Biometrika (`lan1983discrete`{.plain}): error spending at the looks' actual information fractions. `src/bnb/league/sequential.py`.
- **Jennison and Turnbull** (`jennison1999group`{.plain}): the canonical joint distribution of the looks (chapter 3). `src/bnb/league/sequential.py`.

## Model architecture (Yemong)

- **Griffin / RG-LRU**, De et al. 2024 (`de2024griffin`{.plain}): the temporal block (RG-LRU with a causal convolution) and its decay initialisation. The same paper introduces Hawk, which the project does not use. `src/bnb/models/yemong/temporal.py`, `kernels/`; v1 README and `docs/architecture.md`{.plain}.
- **GRU**, Cho et al. 2014, EMNLP (`cho2014learning`{.plain}): the challenger temporal block in arm A16 (`GRUBlock`). `src/bnb/models/yemong/temporal.py`.
- **Transformer**, Vaswani et al. 2017, NeurIPS (`vaswani2017attention`{.plain}): pre-norm attention over a set of tokens. `src/bnb/models/yemong/spatial.py`.
- **RMSNorm**, Zhang and Sennrich 2019, NeurIPS (`zhang2019root`{.plain}): the normalisation throughout. `src/bnb/models/yemong/layers.py`.
- **GLU variants**, Shazeer 2020 (`shazeer2020glu`{.plain}): the gated MLP, GEGLU form. `src/bnb/models/yemong/layers.py`, `kernels/gated.py`.
- **RoPE**, Su et al., Neurocomputing 2024; arXiv 2021 (`su2024roformer`{.plain}): rotary phases, taken here from world x, y and attitude. `src/bnb/models/yemong/layers.py`, `geometry.py`.
- **Gated attention**, Qiu et al. 2025, NeurIPS (`qiu2025gated`{.plain}): the optional sigmoid gate on head outputs (`YemongConfig.gate`). `src/bnb/models/yemong/spatial.py`, `explanation/yemong.md`.
- **Set Transformer**, Lee et al. 2019, ICML (`lee2019set`{.plain}): v1's attention pooling for the outcome value (TeamPMA). v1 `docs/architecture.md`{.plain}.
- **LeJEPA**, Balestriero and LeCun 2025 (`balestriero2025lejepa`{.plain}): cited in v1's training notes. v1 `docs/training.md`{.plain}.

## Value targets and input codes

- **HL-Gauss**, Imani and White 2018, ICML (`imani2018improving`{.plain}): the histogram loss with a Gaussian target. `rewrite/yemong-review.md`.
- **Stop Regressing**, Farebrother et al. 2024, ICML (`farebrother2024stop`{.plain}): HL-Gauss with sigma 0.75 bin widths for the critic and the input codes. `src/bnb/models/yemong/codes.py`, `critic.py`, `explanation/learner.md`, `explanation/yemong.md`.
- **DreamerV3**, Hafner et al. 2025, Nature; arXiv 2023 under a different title (`hafner2025mastering`{.plain}): symlog and two-hot targets. `src/bnb/models/yemong/codes.py`; v1 `docs/architecture.md`{.plain}.
- **MuZero**, Schrittwieser et al. 2020, Nature (`schrittwieser2020mastering`{.plain}): an invertible squashing transform with a two-hot value. `rewrite/yemong-review.md`.

## Search, sampling and systems

- **CMA-ES**, Hansen and Ostermeier 2001 (`hansen2001completely`{.plain}) and Hansen's tutorial (`hansen2016cma`{.plain}): the ship-design search, with (μ/μ_w, λ) weights. `explanation/game/ship-design.md`, `bench/ship_design_search.py`.
- **Sobol sequences**, Sobol' 1967 (`sobol1967distribution`{.plain}): the planned balance sampling study. `rewrite/arms-todo.md`.
- **PyTorch 2 / torch.compile**, Ansel et al. 2024, ASPLOS (`ansel2024pytorch`{.plain}): the compiled rollout, the learner, and the play shell's AOTInductor library. `src/bnb/runtime/`.
- **Triton**, Tillet, Kung and Cox 2019, MAPL (`tillet2019triton`{.plain}): custom kernels (the RG-LRU scan, gated MLP). `src/bnb/models/yemong/kernels/`.
- **Colour-vision simulation**, Machado, Oliveira and Fernandes 2009, IEEE TVCG (`machado2009physiologically`{.plain}): palette checks for colour-vision deficiencies. `rewrite/visual-mockups/README.md`.

## Flight and aircraft performance

- **Energy–maneuverability**, Boyd, Christie and Gibson 1966 (`boyd1966energy`{.plain}, unverified): the lens for the ship card, where the energy store stands in for altitude and the metrics are specific excess power and sustained and instantaneous turn. `rewrite/sections/ship-card.md`, `rewrite/physics-checks/`.
- **NACA Report 868**, Toll 1947 (`toll1947summary`{.plain}): the source of the WWII roll-rate data in `explanation/game/real-scale-research/aircraft.json`. Not to be confused with Gilruth and Turner's NACA Report 715 (1941; `gilruth1941lateral`{.plain}).

## Considered and not cited

The following topics have no matching use in the repository, so no entry was added: TrueSkill, MAP-Elites, CMA-ME / CMA-MAE, pyribs, UMAP, t-SNE, Hide-and-Seek, entity-based policies for variable team sizes, Pocock boundaries, and CUDA graphs (a CUDA feature with no paper to cite). The papers in `rewrite/sections/particles-notes.md` were left out because that survey did not lead to code.

## Unverified

- `boyd1966energy`{.plain}: no primary record (DTIC) was found. Wikipedia and a Marine Corps University archive give *Energy-Maneuverability*, Air Proving Ground Center report APGC-TR-66-4, Eglin AFB, 1966, in two volumes. One source dates the work to 1964, and Gibson's full name is not confirmed.
