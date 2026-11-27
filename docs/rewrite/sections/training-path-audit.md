# Training-path audit: the update, the policy forward, the rollout bookkeeping (2026-11-27 evening)

Read-only audit of the training hot path outside the game tick, at deflector-lookahead 1ec480a3:
the observation and the seat views, the Yemong policy (encoder, trunk, heads), the rollout's
bookkeeping and store, and the update (target pass, micro-batch forward, backward, optimizer).
Companion to `hot-path-audit.md` (the game tick, commit 214f267b); its findings are referenced
by number ("tick audit item 1") and not repeated. Proposals only; nothing here is changed.

Method. The trainer's own code and compile: `lean.learn.compiled.graphs()` (every graph through
`compile_dynamic`), `collect(..., compiled=True, store=...)` and `update(..., compiled=True)`
as `lean.train.run_stage` calls them, with LearnConfig's defaults (2,048 environments, 4 segments
x 128 ticks, 4 epochs, 32 minibatches, micro-batches of 64 views, bf16, streamed store), the
default model (`ModelConfig.for_arena`, 2.16M parameters) and the training game
(`GameConfig(training_spawn=True)`: 4v4, 10 fields, fog). The phases were timed by wrapping the
compiled graphs and the store's methods from outside (CUDA events on the compute stream, so a
phase's time includes its idle gaps, plus host perf_counter), over BC update 1 (update 0 holds
the compiles), then torch.profiler over 8 ticks and 8 micro-batches of BC update 2. RL: the same
collect with the league's networks (a snapshot forced after the first update by
`league_snapshot_steps=(1_000_000,)`: 2 network slots, W = 427 rows each); the RL update is the
same compiled graph with other weights, not timed separately. Then two short A/B jobs: the
micro-batch step (forward + backward) under other compile settings and another scan, and the
tick's parts each compiled alone. Laptop 4070 (8 GB), performance profile, load average 0.05 to
2.0 at the job starts, the GPU lock held per job (another agent's tick jobs between mine, never
concurrent). Scripts and raw output in the scratchpad, not committed. "Measured" means one of
these jobs; "estimate" says what it is computed from.

## Where an update's time goes (BC update, measured)

One update: 158.0 s wall (collect 16.4 s, learn 141.6 s), 6,637 environment steps per second
(the worklog's 2026-11-26 parity row: 6,559).

| stage | seconds per update | share | notes |
|---|---|---|---|
| game sim (`advance`: game.tick, rewards, hit-label scatter, next-state target, redeal) | 1.93 | 1.2% | 3.8 ms a tick; the tick audit's items 1-3 live here |
| observation, seat views, live policy forward, scripted teacher (`act`) | 14.36 | 9.1% | 28 ms a tick; split below |
| league side (`league_actions`) | 0.00 (BC); 3.17 (RL) | 0% / +16% of collect | RL: 2 slots x 427 rows, 2.9 ms a call |
| store writes (device to pinned, copy stream) | 0.04 GPU, 0.15 host | 0.0% | overlapped |
| segment set-up (deals, deal table, carry) | 0.02 | 0.0% | |
| targets and advantages (`target_pass`, stored values, eager) | 0.58 | 0.4% | |
| micro-batch loading (host gather, H2D, widen) | 0.29 GPU (2.2-3.1 host, overlapped) | 0.2% | compute stream never waited (stall 0.00 s) |
| deal-table look-ups (`store.observation`) | 0.02 | 0.0% | |
| update forward (`minibatch_loss`, compiled) | 55.81 | 35.3% | 72.7 ms per micro-batch, 768 an update |
| backward (compiled) | 84.98 | 53.8% | 110.6 ms per micro-batch |
| optimizer step (fused Adam) + clip + zero_grad | 0.035 | 0.0% | 128 steps |
| logging, read-back, averages | < 0.01 | 0.0% | one `.tolist()` an update |
| RL collect total | 19.9 (BC 16.4) | | the RL update is the same graph |

So 89% of the wall time is the micro-batch step (forward + backward), and the GPU is busy 99.5%
of it (profiler: 1,462.8 ms of kernels in a 1,469.6 ms window). Within it (8 micro-batches,
183 ms each, kernels by family):

| family | share of the step | |
|---|---|---|
| GEMMs | 22.9% | |
| GELU / gated MLP (forward and backward) | 15.4% | |
| other pointwise (rotary cat/slice, casts, residual adds) | 14.1% | |
| RMSNorm (forward and backward) | 13.1% | |
| doubling scan (the RG-LRU recurrence, forward and backward) | 12.1% | item 2 |
| sigmoid gates (RG-LRU gates, attention output gate) | 8.8% | |
| attention (memory-efficient SDPA, forward and backward) | 6.4% | |
| losses (log-softmax, HL-Gauss) | 3.4% | |
| rotary / position code (cos, sin) | 2.1% | |
| H2D copies (the streamed store) | 0.7% of the window, on the copy stream | |

The model is narrow (width 128, 73,728 tokens per micro-batch: 64 views x 128 ticks x 9 tokens),
so it is memory-bound: GEMMs are under a quarter of the step and most pointwise kernels run near
the laptop's ~250 GB/s (e.g. the MLP's gelu-times-up, 84 MB in 0.38 ms). Savings come from moving
fewer bytes per token or computing fewer tokens, not from faster matrix products.

The collect's `act` (28 ms a tick, GPU 91% busy in the window), each part compiled alone at the
same state:

| part (compiled alone, E 2048, 3,072 seats) | ms a tick |
|---|---|
| live policy step (`agent_step`: 3,072 seats x 9 tokens, value decode) | 21.5 |
| scripted agent, every ship of both teams (`scripted.decide`) | 5.5 |
| observe, both teams | 0.19 |
| seat views | 0.10 |
| whole `act` (the trainer's graph) | 27.5 |
| `advance` parts: game.tick 0.93, redeal 2.15 (take_deal and both carries' reset), hit-label scatter 0.61, rewards 0.11, next-state target 0.05 | sum 3.85 |
| whole `advance` | 3.88 |

The parts add up to the whole graphs within 1%: no cross-part fusion is lost or gained by the
split. Dtypes under the trainer's autocast (forward hooks, one step): every Linear runs in bf16;
every RMSNorm returns float32, so the residual stream is float32 throughout (encoder output,
every sublayer's input and output); the RG-LRU's convolution, gates and scan are float32 by
design (temporal.py's comment), the position code and rotary angles float32 by design.

## Ranked list

By measured or estimated gain over effort; 1 and 2 are the only large measured items.

| # | what | gain | effort | risk |
|---|---|---|---|---|
| 1 | Compile the learner's graphs with static shapes (the run is static) instead of `compile_dynamic` | measured: micro-batch step 182.8 -> 168.0 ms (-8.1%), about -11 s an update (-7% of wall) | S | none (same graph; gradients within bf16 rounding) |
| 2 | The RG-LRU recurrence as one linear scan instead of the doubling scan (log2 T rounds of whole-tensor `cat`s, forward and backward) | measured: 182.8 -> 159.4 ms (-12.8%); with item 1, 144.4 ms (-21%): about -30 s an update, ~8,200 steps/s (estimate from the step) | M | within rounding (measured) |
| 3 | Pageable host-to-device copies inside compiled graphs (`torch.tensor(...)` constants): 18 a tick in `act`, 4 a micro-batch, each a `cudaStreamSynchronize` | now small: the collect's GPU is 91% busy in the tick window (9% idle), the update's 99.5%; larger on a faster GPU (estimate) | S | bitwise |
| 4 | `reset_carry` rewrites both recurrent carries (151 MB) every tick to zero about 1 environment in 2,000; Policy.step already takes `reset` | estimate ~1 ms of `advance`'s 3.9 ms a tick (0.5 s an update, 0.3%) | S | bitwise or within rounding |
| 5 | Scripted agent on every ship of both teams every tick, also once only the league's scripted games need it (RL after the teacher term is off) | measured 5.5 ms a tick = 2.8 s an update (1.8%); about half of it in late RL with a fixed league slice | S-M | bitwise |
| 6 | The teacher's record: 1,008 MB of the 3.0 GB store, the 42-way joint on all N rows, though it is flight (21) x fire (2) and only own rows are read; still stored, copied and scored when its weight is 0 | memory -0.75 GB pinned, -3 GB H2D an update; time small (the copies overlap) | S-M | within rounding (float16 factors) |
| 7 | Heads and losses on every slot (8) where only own rows (at most 4) are trained | estimate 4-6% of the step (heads ~14% of the model's GEMM MACs, the losses 3.4%) | M-L (needs each team in a fixed half of the slots) | bitwise for own rows |
| 8 | Float32 residual stream: every RMSNorm returns float32 under autocast, so residual adds, norms and their backward move 4-byte values | estimate 5-10% of the step (norms 13.1%, pointwise 14.1%); not measured | S to try | a numerics decision, not rounding |
| 9 | League dispatch: the table width W and the number of slot calls change with the members; `slot_actions` compiles mid-run | measured: the first RL collect with a network slot 64 s against 20 s (~45 s compile); 2.9 ms a slot call at W 427 | M | none |
| 10 | Field memory (encoder field path, map norm and key/value in 4 sublayers, field rotary) recomputed for every tick, though fields and offset are fixed per episode | estimate 3-6% of the step (10 field tokens beside 9 trunk tokens, ~6% of the GEMM MACs plus their cats) | L | within rounding |


## 1. Static compile for the learner's graphs (lean/contract.py:93, lean/learn/compiled.py:21)

`graphs()` compiles act, advance, slot_actions, minibatch_values and minibatch_loss through
`compile_dynamic` (`dynamic=True`, duck sizing off): every size (E, N, F, T, B, seats, league
rows, W) is a symbol. The tick audit's item 1 measured what that costs the game step (looped
ship reductions); the model graphs pay too. A/B on the trainer's micro-batch shape (T 128, B 64,
4v4; lossab.py: the same micro-batch, 3 warm then 20 timed forward + backward):

| compile | ms per micro-batch | loss | gradient norm |
|---|---|---|---|
| `compile_dynamic` (training) | 182.80 | 4.362662 | 1.158940 |
| `torch.compile(dynamic=False)` | 168.02 | 4.362662 | 1.158912 |

The 182.8 ms matches the profiled training step (183 ms), so the A/B shape is the trainer's. The
largest gradient difference is 2.4e-4 against a largest gradient of 6.4e-2, the same for every
variant here: bf16 rounding from a different kernel schedule. Simpler version: compile the
learner's graphs static (`dynamic=False`, or `mark_static` on the fixed axes); a run is fixed in
E, N, F, T, B, so it is one compile per graph per run, and evaluation keeps its own compile. The
act graph (the same model at T 1) should gain in proportion (not measured). Effort S. Risk none.

## 2. The doubling scan (lean/model/temporal.py:29, 142)

`doubling_scan` computes the RG-LRU's h_t = a_t h_(t-1) + b_t by recursive doubling: 7 rounds at
T 128, each a `torch.cat` of a slice and a shifted product over the whole (T, B, M, width) float32
tensor (37.7 MB a micro-batch), and the decay's own doubling; autograd keeps every round. In the
profile the scan's kernels are 12.1% of the micro-batch step (forward ~5 ms, backward ~17 ms),
O(T log T) work for an O(T) recurrence. Tried here (experiment only, patched in the A/B process):
a custom autograd Function whose forward is `torch._higher_order_ops.associative_scan` (one
inductor scan kernel) and whose backward is the reversed scan (dL/dh_t = g_t + a_(t+1) dL/dh_(t+1);
d decay = dL/dh x h_(t-1), d update = dL/dh):

| compile | doubling scan | linear scan |
|---|---|---|
| `compile_dynamic` | 182.80 ms | 159.42 ms |
| `dynamic=False` | 168.02 ms | **144.36 ms** |

Same loss to 7 digits, gradients within the bf16 difference above. Caveats: `associative_scan`
is a prototype (no autograd of its own, hence the Function; "may miscompile"), so a real version
wants a test against the doubling scan at T 1, 2, 128 with resets, or a small hand-written Triton
kernel (sequential over T, one thread per channel: 73,728 channels of 128 steps). Effort M. Risk
within rounding. With item 1 the step is 21% faster: 768 steps x 38.4 ms = 29.5 s an update.

## 3. Host waits inside the compiled graphs (lean/learn/returns.py:18-24, lean/model/policy.py:221, lean/scripted/paths.py:105-108, 196, 309, lean/scripted/opponent.py:146-147, 778)

`value_centres` and `next_state_centres` build their tables with `torch.tensor(...)` on the host
and `.to(device)`; the scripted agent builds small index and time tables with
`torch.tensor(..., device=...)`. Inside a compiled graph each becomes a pageable host-to-device
copy at every call, and a pageable copy makes the host wait for the stream
(`cudaStreamSynchronize`). Counted in the profiles: 18 copies a tick in `act` (12 to 1,260
bytes), 4 a micro-batch in `minibatch_loss` (the value and next-state centres and knees). So the
host never runs ahead of the GPU: `act`'s host time (30 ms a call) equals its GPU span, and the
loss call's host time (174 ms) contains the previous backward. Today it costs little because the
GPU is the bottleneck (the collect's window 91% busy, the update's 99.5%), but on the 5090 the
host's launch time per tick stays and the kernels shrink (estimate: then it is the limit).
Simpler version: build the tables once on the device and pass them in (graph inputs, or
module buffers); `torch.tensor(..., device=cuda)` inside the graph is not enough, it is still a
host constant copied at every call (the scripted agent's copies are of that form). Effort S. Risk bitwise.

## 4. reset_carry rewrites both carries every tick (lean/learn/rollout.py:447-448, lean/model/agent.py:52)

`redeal` multiplies the live policy's carry (2 blocks x 3,072 seats x 9 tokens x 4 x 128, float32,
113 MB) and the opponents' (38 MB) by `~done` every tick; about one environment in 2,000 ends in
a tick. `redeal` compiled alone is 2.15 ms of `advance`'s 3.88 ms (take_deal's share is the tick
audit's item 3). Policy.step already takes `reset` and Temporal zeroes the carry's contribution
through `fresh` (the update's sequence does exactly that from the stored `reset`), so the
rollout can pass `carry.reset` (already recorded) to the step and drop the multiply. Estimate
~1 ms a tick (two 151 MB passes at ~250 GB/s), 0.5 s an update. Effort S. Risk bitwise or
within rounding (the zeroed state enters as `fresh` 0 instead of a zero carry; `carried` reads
the stale carry but `continuing` is 0 there).

## 5. The scripted agent on every ship every tick (lean/learn/rollout.py:542)

`act` runs `scripted.decide` on all E x N ships: 5.5 ms a tick (measured alone), 2.8 s an update.
In BC that is right (the teacher of every own row, and the league's scripted games). In RL the
teacher term switches off (gate 0.45 or weight below 0.005) and from then on only the league
environments playing member 0 read it; the self-play half (E_s environments) never does. Simpler
version, static: once the term is off, run it on the league slice only (a fixed E - E_s rows;
rows playing networks are computed and unused, as today). About half the cost in late RL.
Effort S-M (one more compiled variant of `act`, switched once). Risk bitwise.

## 6. The teacher's record (lean/learn/rollout.py:595, lean/learn/store.py)

The store's largest record: (T, V + 1, N, 42) float16 = 1,008 MB of the 3.0 GB of per-tick
records (ship features 552 MB, carry 432 MB, value and reward 240 MB each). It is the joint
`flight x fire` (lean/scripted/opponent.py:656: 21 flight probabilities times a Bernoulli fire),
stored for all N slots of the seat, of which the loss reads the own rows. Stored as 21 + 1
numbers on own rows: about 264 MB. In RL with the teacher off it is still recorded, copied to
the device every epoch (4 GB of H2D an update), widened to float32 and scored with weight 0.
Time: small here (the streamed copies overlap, the compute never waited); memory: 0.75 GB of
pinned host memory and the staging buffers, which matter for the all-on-device option on the
5090 (`stream_rollout=False`). Effort S-M. Risk within rounding (the product of float16 factors).

## 7. Heads and losses on enemy and empty rows (lean/model/policy.py:604-608, lean/learn/training.py:174-202)

The heads run on all N = 8 ship tokens of a seat; the losses then mask to `own` (at most 4, the
seat's living ships). Per row the heads are ~319k MACs (head and auxiliary first layers 2 x 65k,
the last layers 42 + 315 + 378 + 3 outputs at 256): about 14% of the model's GEMM MACs (estimate
from the layer sizes; trunk ~1.66M MACs a token). Their float32 outputs (738 a row) feed
log-softmax and HL-Gauss cross-entropies (3.4% of the step, measured) on every row. Half of it
is enemy rows (estimate 4-6% of the step). A fixed-shape version needs a fixed team layout: the
deal puts team 0 in the first `team_sizes[:, 0]` slots (core.py:2717), which is the first N/2
only when teams are full. With each team in its own half of the slots (absent ships inside
it; the model has no slot-position input, so this is invisible to it), a seat's own rows are a
fixed slice, and the heads, losses and the per-row records (teacher, next-state, action, log
probability, hit label) shrink to N/2. Effort M-L. Risk bitwise for own rows.

## 8. The float32 residual stream (lean/model/policy.py, temporal.py)

Hooks under the trainer's autocast: every `nn.RMSNorm` returns float32, so the encoder's output
and every sublayer's residual are float32; each Linear casts its input to bf16 (fused into the
norm kernels). RMSNorm forward and backward are 13.1% of the step, the remaining pointwise
14.1%. The comments say the position phases and the recurrence stay float32 by design; nothing
says the residual should. A bf16 residual (norms computing in float32, returning bf16) halves
those bytes; estimate 5-10% of the step, not measured. This is a numerics decision (bf16
residual streams are common; float32 ones are the safer default), so it is an experiment to
run, not a clean-up. Effort S to try.

## 9. League dispatch (lean/learn/pool.py:362-388, lean/learn/rollout.py:346)

`League.prepare` sets the table width W = max(ceil(1.25 L / members present), rows a slot holds)
and the list of active slots before every segment; `network_actions` then calls `slot_actions`
once per active slot (1 to 18) on W rows. Both change during a run (W 427 with 3 members,
86 with 15; slots 2 to 18), which is why the graph must be dynamic, and the first segment with a
network slot compiles it mid-run: RL collect 1 took 64 s against 20 s for the next (measured).
Steady cost: 2.9 ms a slot call at W 427 (3.2 s an RL update with 2 slots, +16% of the collect).
Fixed-shape version: one table of the maximum slots (2 + 12 + 4) x a fixed W, the networks
stacked and run in one batched call (torch.func.vmap over stacked weights; D5 had this,
core-worklog 05:58), empty slots carrying zero rows. Effort M. Risk none (same moves).

## 10. Field memory every tick (lean/model/policy.py:558-570, Spatial.forward)

Fields and the position offset are fixed within an episode, yet every tick of every view runs
the field token's first projection, the shared Linear, the map memory, and in each of the 4
spatial sublayers the map norm, the map key/value projection, the memory rotation and the `cat`
of its keys and values to the tokens'. 10 field tokens beside 9 trunk tokens (they never pass
the MLPs, so ~6% of the GEMM MACs, plus norms, rotations and cats). Simpler version: compute
the per-sublayer field keys and values once per deal-table row (the store already keys the
fields by deal row) and gather them per tick. Estimate 3-6% of the step. Effort L (the
gather has to stay differentiable; the rollout's step needs the same cache per environment).
Risk within rounding.


## Dynamic shapes in training (the author's rule: every one is a bug)

Found in the training path (a run fixes E, N, F, T, B, segments and lanes):

1. Every learner graph compiled with dynamic sizes (`compile_dynamic`, lean/contract.py:93,
   used by lean/learn/compiled.py:21-27 for act, advance, slot_actions, minibatch_values,
   minibatch_loss). Nothing in a run changes these sizes except item 2, so the dynamism buys
   nothing in training and costs 8% of the update step (item 1) and the game's looped
   reductions (tick audit item 1).
2. The league's network table (lean/learn/pool.py:370-388): W follows the members present
   and the rows held, and the number of slot calls per tick follows the active slots
   (rollout.py:346). A real shape change during a run; it compiles `slot_actions` mid-run
   (item 9).

Checked and static: the slot count N (start deals at the largest team total, rollout.py:377;
variable team sizes only move team 1's first slot, a value not a shape); the store's
micro-batches (padded with the spare view, store.py:298-327) and the target pass's chained
rows (the same B, 64); NextEpisodes (P = E); the hit-label buffer (2 T E N + 1); F (fixed by
`field_count`; the F = 0 branches are per run); the final bootstrap `act` (same shapes);
`abandon` (a full E deal). No data-dependent shape in lean/learn, lean/model or the scripted
call (no nonzero, masked_select or boolean indexing). No graph breaks or recompiles in the
trainer's graphs: one compiled region per call for act, advance, the loss forward and its
backward in the profiles, and none logged (TORCH_LOGS=graph_breaks,recompiles over the tick
parts). One trap seen while measuring, not a bug in training: `act` recompiles if the carry
requires grad (collect runs under no_grad).


## Belief state

None outside the Griffin layer. `lean.model.policy` says "Yemong trunk without beliefs", and a
search of lean/ for belief, predictive and world-model code finds only that docstring. The carry
is the RG-LRU state and the convolution's three-tick buffer (Temporal), nothing else. Two things
that are not belief state but sit near it: the game's fog record (`State.seen`,
`ticks_since_seen`: what a team last saw of an enemy, a game rule the observation reads) and the
next-state and hit heads (auxiliary losses on the trunk, no state carried; the rollout records
their targets, 144 MB of next-state targets an update).

## Looks wasteful, is cheap (measured)

- The optimizer: fused Adam, 0.23 ms a step; the global clip (foreach norm) 0.05 ms; zero_grad
  with set_to_none. 0.035 s an update together.
- The streamed store: the host gathers each micro-batch from pinned memory (2.2-3.1 s of host
  time an update) and copies it on a side stream; the compute stream waited 0.00 s (measured
  with the store's own stall events); the copies are 0.7% of the window and overlap. Keeping
  it all on the device (`stream_rollout=False`) would not be faster here; it matters only for
  memory.
- The target pass (GAE over 512 chained ticks, eager, stored values): 0.58 s an update.
- The two `observe` calls and the seat views: 0.19 and 0.10 ms a tick; the gathers a reshape
  could replace (index_select by seat) are cheap at (S, N, ~35) floats.
- Widening the compact store on the device (float16 to float32 casts per micro-batch,
  `store.batches`): within 0.29 s an update. The model then casts to bf16 in-graph; passing
  float16 directly saves a little more, not worth a rule.
- HL-Gauss targets rebuilt every micro-batch and epoch (value 5 x 63, next-state 6 x 63): the
  erf and the cross-entropy fuse into the loss kernels; all losses together are 3.4% of the step.
- The teacher's own entropy (for `bc_kl`) recomputed every micro-batch: part of the 3.4%.
- The additive attention mask built per forward, `next_state_target` (0.05 ms a tick), rewards
  (0.11 ms), logging (one read-back an update).
- Value logits in the rollout step: needed (stored values for the targets); the auxiliary
  heads are already skipped there (`with_auxiliary=False`).
- The live policy recomputing what the rollout computed: the update's forward cannot reuse the
  rollout's (the weights change every minibatch), apart from the first minibatch of epoch 0.


## Not audited

The scripted agent's internals beyond the measured total and the pageable copies (its own audit:
the lookahead branch); the evaluation runner; memory use beyond the peak (5.72 GiB allocated).
The RL update was not timed (same graph as BC with other weights; the teacher term's weight
only). Numbers are from one laptop GPU; the 5090 shifts the balance towards the host-side items
(5, 9).
