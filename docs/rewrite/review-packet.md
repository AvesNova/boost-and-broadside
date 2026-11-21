# Review packet: the plan to rebuild boost-and-broadside on a lean core

Prepared 2026-11-21 for an independent review. The author of this packet also wrote the plan
it describes, so its selection and framing carry that author's view. Section 4 labels each
claim by how it is known, and section 5 lists what was not checked. The repository branch
`simplify/audit` holds everything referenced here.

## 1. What is being reviewed

The project is a 2D air-combat game with reinforcement-learning agents. Its owner (here,
"the author") wants three things at once:
- a game people play and find fun, against trained agents, in real time;
- experiments that answer research questions, possibly a paper;
- cheap training on one GPU.

**Version history:**
- **v1** (branch `main`, about 33k lines of source) trained a recurrent attention policy
  with PPO.
- **v2** (branch `v2`, about 63k lines of source, 45k lines of tests, about 52k lines of
  design notes) was a rewrite meant to be faster and more general. It is slow to compile
  (minutes per stage on a laptop, about an hour per stage on rented GPUs), very slow
  uncompiled, and not clearly faster than v1 in steady training.

**The plan:** build a minimal "lean core" (a new game step, policy and PPO meeting outcome
requirements), then port v2's features into it one at a time, each justified by
measurement or by the scope. v2 stays as a read-only reference.

**Documents:**
- `docs/rewrite/core-brief.md`: the operating brief for the agents that would do the work
  (requirements, strategy, milestones, authority, measurement, records, writing and test
  rules, assumptions).
- `docs/rewrite/simplification.md`: the audit behind it (measurements, the inventory of
  workarounds, root causes).

## 2. What the review is asked for

A ranked list of objections and risks, most consequential first. For each:
- what is wrong or missing;
- the consequence if it is not addressed;
- the evidence or reasoning;
- what observation would show the objection is wrong.

Then, briefly, what you agree with and why, and anything the plan does not consider at all.

Specific questions:
1. **Diagnosis.** Are the root causes in section 3 the right ones? What is misattributed
   or missing?
2. **Strategy.** Is a lean core plus earned additions the right approach, against
   simplifying v2 in place? How would this third rewrite fail?
3. **Requirements and instruments.** Will hard limits on outcomes, with proxies tracked
   under a ratchet rather than capped, produce the intended behaviour in coding agents?
   What remains gameable?
4. **Agent process.** One lead agent, short-lived helpers, the authority rules, the record
   formats, verification in proportion to stakes. Where will it break?
5. **Technical bets** (brief, section 10):
   - PyTorch with dynamic batch and ship dimensions;
   - no belief model for fog, only a recurrent trunk;
   - plain PPO and one value head;
   - shot labels recorded by the game;
   - positions in metres, never normalised by the arena;
   - modes mixed in one batch, as contiguous slices;
   - real-time play on the CPU.
6. **What is missing entirely.**

**Budget:** do not re-derive the measurements. Spot-check up to five claims of your choice
against the scripts or the code if you doubt them. Section 4 says where each claim comes
from.

## 3. The diagnosis, as written in the audit

Underneath about 80 identified workarounds, the audit names five root problems:
1. **No budgets or outcome measures.** Cost accumulated unseen; each symptom got a local fix.
2. **The representation.** Inputs and outputs are hand-built categorical codes over very
   wide ranges, written per component in Python loops. This is where most operations, the
   decoding, the label machinery, the reward scaling and the precision fixes come from. The
   trunk itself is small.
3. **Static shapes and graph capture treated as fixed rules,** inherited from an early plan
   to use JAX. Consequences: every branch computed and then selected, constants as tensors,
   a recompile per shape, and the configuration bent to compile once.
4. **Requirements never pruned:** bit-exact reproducibility, backward compatibility,
   generality, options kept on by default.
5. **Optimisation aimed at the rollout (about 10% of an update) and at custom kernels,**
   rather than at the representation and the learner's surroundings.

The audit also names process causes:
- agents with narrow assigned areas;
- working rules that made adding easier than removing;
- decisions recorded without the conditions they depended on;
- tests and notes that pinned implementation details.

## 4. Evidence, by how it is known

### Measured in this session (CPU only; reproducible)

Method: a `TorchDispatchMode` counts every operation PyTorch dispatches in eager mode,
attributed to named regions. View-only operations are excluded. v2's Triton kernels count
as one operation each, as on a GPU. FLOPs come from `torch.utils.flop_counter.FlopCounterMode`.
Scripts and records are in `docs/rewrite/sections/simplification/`.

| Claim | v1 | v2 | Notes |
|---|---|---|---|
| Operations per rollout step (one decision) | 7,574 | 27,019 | Identical at small and full batch sizes |
| Operations per policy forward | 649 | 5,172 | v2: input encoding 1,180; trunk 295; forecast decoding 3,588 |
| Operations per environment step | 2,549 (two physics ticks) | 5,667 (one tick) | |
| v2 correction decode per call | | 3,023 | 87% a Python loop over 14 components |
| Operations per learner micro-batch | about 2,200 | about 17,500 at segment length 128 | v2 value extrapolated linearly from lengths 16 and 32 |
| Matrix-multiply FLOPs per training sample, forward and backward | 169 M | 214 M | v2 includes recomputation inside the kernels' backward |
| Bytes moved per sample, uncompiled | about 23 MB | about 17 MB, plus about 60 MB per view tick once per update for shot re-simulation | Uncompiled counts; compiled code fuses, so ratios matter more than sizes |
| v1 training steps per second (laptop, whole run, W&B export) | 2,670 and 2,707 (medians) | | Includes compile and in-training evaluation |

### Read from v2's own notes (secondhand, under v2's conditions)

- **Compile times:** a ratings act stage of 143 s at 5,623 traced operations; a loss of 64 s
  at 3,162; a play tick of 25.8 s at 2,867. About an hour per stage on rented machines.
  About 50 s warm.
- **Where an update's time goes:** about 90% learner; rollout 16-24 s of 90-250 s. The
  learner is GPU-bound at large micro-batches and host-bound at small ones.
- **The custom kernels** gave 1.23x training speed in bf16 at 160-row micro-batches;
  attention was about 65% of that gain.
- **An uncompiled tick at batch size 1** cost 25-30 ms for about 4,500 operations, about
  6 microseconds per operation.
- **Rating noise:** about 110 Elo between seeds. The error head's measured benefit was about
  +27 Elo (the notes' decision bar was +40 to +50).
- **v1's CUDA graphs** were measured as no gain (`decisions.md`).
- **A JAX against PyTorch "physics race":** no structural difference in steady speed; JAX
  compiled the physics in under a minute.

### Inferred, not measured

- Making batch-like dimensions dynamic costs a few per cent at most.
- Python constants instead of tensor constants change compiled steady speed little and
  uncompiled speed noticeably.
- About 10,000 traced operations is a rough ceiling for a compile of a few minutes (from
  the per-operation rates above).
- A grid-cell-like position code costs about 3-5 operations.
- One policy decision at batch size 1 takes about a millisecond on a CPU once the
  operation count is small, so play can run on the CPU.
- From memory, not verified: JAX has no native Windows GPU support.
- The five root problems and their ranking.
- That a lean core is better than simplifying v2 in place.

### The workaround inventory

Produced by four read-only agent reviews of v2's code and notes. Several surprising claims
were spot-checked against the code and held (for example, that every opponent kind runs on
every opponent row, and that a 5,170-line parameter file is read only by a test). Most
entries were not individually verified.

### Decided by the author (preferences and scope)

These are not open for argument unless they conflict with another goal:
- one GPU; the real-scale ship set only; two teams;
- training robust to team sizes; variable map sizes if they work;
- the playable game in scope; no bit-exactness; no backward compatibility;
- no belief model at first;
- drop the shot trajectory, density and correction heads; keep next state, shot outcome and
  geometry; merge the outcome head into value;
- drop the custom kernels;
- positions in physical units without rescaling per map;
- outcome requirements rather than fixed budgets;
- documentation and test rules as in the brief;
- lean core preferred.

## 5. Not checked, or known to be weak

- **No GPU measurements were made in this session.** Every GPU timing is from v2's notes.
- **Two different operation counts are in use.** Dispatch-level counts (this session's) and
  Dynamo's traced counts (v2's notes) differ, so compile estimates based on per-operation
  rates mix them.
- **The learner's figures at length 128 are linear extrapolations** from two points.
- **The FLOP count** for v2 includes recomputation inside its custom kernels' backward.
- **The cost of dynamic shapes in PyTorch was not measured.** The plan depends on it being
  small.
- **The learning effects of the planned simplifications are unknown.** Removing the belief
  model and the heads may change how fast agents learn. Only the planned learning canary
  will show it.
- **The topic index of v2's notes** was built from headings and sampled passages.
- **Alternatives considered and set aside:**
  - simplifying v2 in place (rejected because the static-shape and code-based design
    reaches every interface);
  - switching to JAX now (deferred to a measured checkpoint after the skeleton);
  - fixed numeric budgets on operation counts (replaced by a ratchet, over concern that
    agents fill a budget);
  - one up-front review only (milestone reviews proposed instead).
