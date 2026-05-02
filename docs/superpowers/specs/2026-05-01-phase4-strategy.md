# Phase 4 — Strategy and Roadmap

**Date:** 2026-05-01
**Branch:** `engine-v2`
**Author:** Chitiiran (with Claude)
**Status:** Planning doc; closes Phase 3 with action items for Phase 4.

This doc captures everything learned during Phase 3 about the v2 engine,
the GNN training results (including the failed and partially-failed
experiments), the published-state-of-art comparison, and concrete next
steps within realistic hardware constraints.

---

## 1. Where Phase 3 left us

### 1.1 What works
- **v2 engine**: full Catan rules (280 actions, dev cards, ports, longest
  road, largest army, player trades). 8 bug fixes shipped during Phase 3
  including the trade-loop wedge, terminal-check on LR/LA bonus, and
  rollout length-discount.
- **Recorder**: per-game parquet flush + atexit + skip_game-keeps-data.
  Sweeps are SIGINT-safe; at most one in-flight game ever lost.
- **Data sweep**: 2000 games × sims=100 × MCTS-vs-random in **75 min**
  (5 workers). 1998/2000 natural completions, **463 848** recorded MCTS
  positions.
- **GNN training pipeline**: end-to-end. Cache 463k positions in 24 min,
  train at ~6.7 min/epoch on GTX 1650, save checkpoints + plots + log.
- **Tournament harness**: e7 4-way (GnnMcts, PureGnn, LookaheadMcts,
  Random) with GPU support. Runs 20 games in 12 min.
- **Hex-rotation augmentation**: full 6-fold permutation tables verified
  by 15 unit tests. Two modes (fixed-k and random-k). Ready for use.
- **Playback viewer**: per-game self-contained HTML. Used to find the
  6892-step pathological game that exposed the rollout-decay bug.
- **Safety harness**: 49 Rust + ~85 Python tests + 5 baseline-replay
  fingerprints + 5 perf workloads + playback render. ALL GREEN through
  Phase 3.

### 1.2 What didn't work
- **Trained GNN inside MCTS**: 0/20 in tournament — *worse* than random
  rollouts. The GNN's learned policy at val_top1=0.273 isn't strong
  enough to redirect MCTS toward winning lines.
- **PureGnn argmax**: 5% in baseline tournament, 0% after rotation
  fine-tuning. Both numbers consistent with the 25%-uniform-baseline
  for 4-player game, i.e. **no learned signal beyond random**.
- **Single-rotation augmentation**: best val_top1 went from 0.273 to
  0.276 (+0.003, within noise). Tournament unchanged.

LookaheadMcts (sims=25, depth=25) wins 90-95% of all 4-player tournament
games. **It's the de-facto strong baseline; our trained models don't
beat it.**

### 1.3 The bottleneck (data-backed)
The bottleneck is NOT engine speed:

| | Pure Rust | Python+PyO3 | Python+MCTS |
|---|---:|---:|---:|
| per game | 2 ms | 7.8 ms | 5800 ms |
| 86.7% of MCTS time → | random_rollout (Rust, called many times per move) | | |

The bottleneck is **training data quality and quantity** combined with
**a fixed-data supervised setup** (we trained on RandomBot-vs-MCTS
games, not self-play).

---

## 2. The data-backed comparison

We surveyed published Catan RL projects to understand the gap between
our setup and a successful one.

### 2.1 Catanatron (Bryan Collazo)

The most popular Catan AI/simulator project. From his Medium post
"5 Ways NOT to Build a Catan AI":

| Metric | Value | Source |
|---|---|---|
| TensorForce vs Random win rate after 8 hours | 80% | Medium post |
| MCTS vs WeightedRandom (sims=25) | 100 games tested | Medium post |
| WeightedRandom vs Random | ~61% in 10000 games | Medium post |
| Total self-play games | not disclosed | — |
| Model architecture | not disclosed | — |
| GitHub repo training data scale | not in README | github.com/bcollazo/catanatron |

**Catanatron itself** (the deployed bot) doesn't publish training
metrics, just a runnable simulator + bots. The Medium post's claim is
"80% vs random in 8 hours" with no full-bot benchmarks.

### 2.2 Henry Charlesworth — Settlers of Catan with Deep RL ([settlers-rl.github.io](https://settlers-rl.github.io/))

The most rigorously documented end-to-end RL Catan project. Hard
numbers from his project page:

| Metric | Value |
|---|---:|
| Training duration | **~1 month** continuous |
| Total decisions processed | **~450 million** |
| Total PPO updates | ~3 500 |
| Parallel processes | 128 |
| Envs per process | 5 |
| Decisions per env per batch | 200 |
| Batch size | 128 000 decisions (128 × 5 × 200) |
| Optimization epochs per batch | 10 |
| Minibatch size | 2 000 samples |
| **Hardware** | **32-core CPU, 128 GB RAM, RTX 3090** |
| Eval games per checkpoint | 1 200 |
| Final 4-way result vs 3 self-copies | **47/100** (random would be 25%, so significant) |

This is the headline state-of-art for "single-person published Catan
RL" as of 2026-05.

### 2.3 Our setup vs Charlesworth

| | Charlesworth | Ours (Phase 3) | Ratio |
|---|---:|---:|---:|
| Training-decisions seen | 450 000 000 | ~7 400 000 (5 ep × 463k) | **60×** more |
| Wall-clock on training | ~1 month | ~1.7 hours | **400×** more |
| Hardware (GPU) | RTX 3090 | GTX 1650 | **~10×** stronger |
| Hardware (CPU/RAM) | 32-core / 128 GB | 8-core / 16 GB | ~4× / 8× |
| Method | PPO self-play | Supervised on MCTS-vs-Random | qualitatively different |
| Tournament against itself+copies | 47% (4-way) | 0% (4-way mixed lineup) | not directly comparable |

**Our gap is real and large**: ~60× less data, ~10× weaker GPU, and a
training paradigm that's strictly weaker than self-play. Beating
Charlesworth head-on isn't realistic with our hardware.

### 2.4 Other projects surveyed

- **QSettlers (Krishna)**: Q-learning, smaller-scale.
- **Re-L Catan (Stanford CS230)**: PDF was unparseable in our fetch;
  numbers not extracted.
- **Asher's "Modeling Catan through self-play"**: blog post; not deeply
  surveyed.
- **Playing Catan with Cross-dimensional NN (arxiv.org/2008.07079)**:
  paper exists; not deeply surveyed.
- **Actor-Critic Catan thesis**: extant.
- **Vombatkere's Catan-AI**: handcoded bots and minimal RL.

None of these match Charlesworth's scale, and none beat handcoded
heuristics by clearly-documented margins.

---

## 3. Honest gap analysis

We have **two structural disadvantages** vs Charlesworth's setup:

1. **Hardware**: ~10× less GPU, ~4× less CPU. Can't be closed.
2. **Training paradigm**: supervised-on-fixed-data vs self-play.
   **Self-play is strictly more sample-efficient** because:
   - Every iteration produces fresh data with the *current* model
   - The model and the data difficulty co-evolve
   - There's no ceiling from "what the random opponents allow"

We have **three asymmetric advantages**:

1. **Engine speed**: pure-Rust v2 random game = **2 ms** vs typical
   Python Catan ~50-200 ms. Per CPU-second we generate **25-100× more
   games** than Python implementations.
2. **Engine completeness**: full Catan rules with bug-fixed termination
   and trade-loop guards. We can trust our data.
3. **Tooling**: per-game salvage, replay viewer, MCTS visit recording,
   safety harness. Faster iteration than re-running from scratch.

The path to a competitive bot uses our advantages and avoids head-to-
head competition with Charlesworth's compute scale.

---

## 4. What we tried in Phase 3 and what we learned

### 4.1 Phase 3.4 — Supervised training data (MCTS vs Random)
- 2000 games × sims=100, 463 848 positions, 75 min wall clock.
- **Healthy data**: 99.9% natural completions, MCTS winrate 82%,
  median game 1109 steps.
- **Limit**: training set ceiling = "what an MCTS bot does against
  random opponents." Doesn't include competitive lines.

### 4.2 Phase 3.5 — Baseline GNN training
- 80k params, hidden_dim=32, 2 SAGEConv layers.
- Best val_top1 = **0.273 at epoch 4**, plateaued thereafter.
- Train_loss kept decreasing while val_loss climbed (classic overfit).
- We early-stopped at epoch 16/20.

### 4.3 Phase 3.6 — Tournament (GnnMcts, PureGnn, LookaheadMcts, Random)
- First attempt CPU-only: all 20 games timed out (GNN forward too slow
  on CPU at sims=25).
- After GPU patch: **20/20 natural completions, 12 min wall**.
- Result: **LookaheadMcts 90%, PureGnn 5%, Random 5%, GnnMcts 0%**.
- The trained GNN is *worse* than random rollouts inside MCTS.
- 1 PureGnn win in 20 is at the noise floor.

### 4.4 Phase 3.5b — Single-rotation fine-tune
- 5 epochs from baseline-checkpoint with **fixed 60° rotation**.
- Best val_top1 = 0.276 (+0.003 vs baseline).
- Tournament unchanged: GnnMcts 0/20, LookaheadMcts 19/20.
- **Conclusion**: the message-passing GNN already has implicit
  rotation invariance from local conv layers. Fixed-rotation
  augmentation adds near-zero new gradient signal.

### 4.5 Phase 3.5c — Random-rotation fine-tune (in flight)
- 5 epochs from rot60-checkpoint with **random k uniformly in 0..5**.
- Tests whether forcing every batch to see different orientations
  produces meaningfully better generalization.
- Expected: small improvement but not enough to flip the tournament.

---

## 5. Phase 4 strategy

### 5.1 The thesis

**Don't try to match Charlesworth's compute.** Use his target as a
ceiling and our advantages as the path.

Three asymmetric wins to chain:

1. **Faster, simpler rules during training** — strip player trades and
   most dev cards. Re-enable for final benchmark.
2. **Stronger teacher via imitation** — train the GNN to imitate
   LookaheadMcts (the 90%-winrate handcoded bot), not MCTS-vs-random.
3. **Short MCTS-self-play loop** — once we have a decent imitation
   model, switch to AlphaZero-style self-play for ~5-10 iterations.

Each is testable independently. Each compounds.

### 5.2 Concrete roadmap (~3 weeks part-time)

#### Week 1 — Engine speed + simplified rules
- Add `--simplified` engine flag:
  - Disable `Action::ProposeTrade` entirely (action space 280 → 260)
  - Disable dev-card buy + 5 dev-card play actions (→ ~225)
  - Optionally: cap game length at e.g. 500 steps with VP-margin tiebreak
- Bench: ~3-5× faster random game expected.
- Re-baseline: re-run Phase 3.4 sweep with `--simplified` (target ~30 min for 2000 games).
- Re-baseline: re-train GNN on simplified data (target ~30 min).
- Tournament with simplified rules. Goal: confirm pipeline still works,
  see if val_top1 improves with smaller action space.

#### Week 2 — Imitation training (LookaheadMcts as teacher)
- Generate a sweep where **all 4 players** are LookaheadMcts (sims=25,
  depth=25). The recorded "action distribution per move" is now the
  teacher signal.
- 1000 games × all-LookaheadMcts × 4 workers ≈ ~30-60 min.
- Train GNN to predict LookaheadMcts visit distribution (same loss
  function as before, just different teacher).
- Tournament: PureGnn-imitation vs LookaheadMcts vs Random.
  - **Target winrate: ≥25%** (matches uniform baseline, beats Random).
  - **Stretch: 35-50%** (close to LookaheadMcts itself).

#### Week 3 — Short MCTS-self-play
- Initialize with the imitation-trained GNN.
- Loop:
  1. Run 200-500 games of GnnMcts (sims=50) self-play (4 GnnMcts
     copies). Record visit distributions.
  2. Train 2-3 epochs from current weights on the new data.
  3. Tournament: new GNN vs previous GNN. If new wins ≥55%, accept.
- Expected: 5-10 iterations × ~30-60 min each = **~6 hours total**.
- This is the AlphaZero core loop, scaled to our hardware.

#### Week 4 — Re-enable full rules + final benchmark
- Re-introduce trades + dev cards in the engine.
- Fine-tune the trained GNN on full-rules data for ~5 epochs.
- Final tournament with the full lineup. Target: GnnMcts beats
  LookaheadMcts in some seedings.

### 5.3 Estimated costs at each stage

| Stage | Wall clock | GPU hours | New artifacts |
|---|---:|---:|---|
| Week 1 (simplified rules + retrain) | ~3 hours | ~0.5 | simpler engine, refreshed dataset, refreshed baseline |
| Week 2 (imitation) | ~3 hours | ~0.5 | imitation GNN, beats Random |
| Week 3 (self-play loop) | ~6 hours | ~3 | progressively-stronger GNNs |
| Week 4 (full rules + finetune) | ~3 hours | ~0.5 | final GNN |
| **Total** | **~15 hours** | **~5** | |

That's a single-weekend workload, not a month. The reduction comes
from imitation > supervised-on-random, simplified rules > full rules
during training, and self-play > fixed teacher.

### 5.4 Things we should NOT pursue

- **Bigger model**: 80k params is already enough for Catan's state
  size. Going to 500k+ would slow training, increase overfitting risk,
  and add nothing useful at this dataset size.
- **More fixed-data supervised training**: the val_top1 ceiling we hit
  (0.27) is the dataset ceiling. More epochs over the same RandomBot
  data won't break through.
- **6-fold rotation augmentation alone**: we proved single-rotation
  doesn't help. The implicit rotational equivariance of the message-
  passing GNN already captures most of what 6-fold augmentation would
  give. Random-rotation augmentation may help a bit (in-flight test).
- **PPO/policy gradient at our compute scale**: too sample-inefficient.
  Charlesworth needed 450M decisions; we'd need similar or our model
  collapses.
- **Matching Charlesworth's compute**: we can't. Don't try.

### 5.5 Key risks

1. **Imitation might not transfer**: the GNN might learn to mimic
   LookaheadMcts on observed data but fail in unseen states. Mitigated
   by including diverse seeds + opponents in the imitation sweep.
2. **Self-play loop might collapse**: classic AlphaZero failure mode
   where the model finds a degenerate self-play equilibrium that
   doesn't generalize. Mitigated by the per-iteration tournament gate
   (only accept new GNN if it beats the old one ≥55%).
3. **Simplified-rules transfer might be weak**: a model trained on no-
   trades / no-dev-cards Catan might not adapt to full rules in a
   single fine-tune pass. Mitigated by Week 4's longer fine-tune
   budget.
4. **Single-machine compute might still bottleneck**: if Week 3's
   self-play takes 6 hours per iteration instead of 30 min, the loop
   becomes infeasible. Mitigated by simplifying rules first.

---

## 6. Engineering work needed (small)

The following code work supports the strategy:

| Item | Effort | Where |
|---|---:|---|
| `--simplified` engine flag | ~1 hour | `catan_engine/src/state.rs` + `rules.rs` |
| LookaheadMcts-as-teacher recorder | ~1 hour | new `e10_lookahead_imitation.py` experiment |
| Self-play tournament gate | ~2 hours | new `e11_selfplay_loop.py` experiment |
| Win-margin / step-discounted value target | ~1 hour | `dataset.py` value-target computation |
| Resume self-play loop from any iteration | ~1 hour | extend `train.py --resume` semantics |
| Multi-perspective augmentation | ~2 hours | `state_to_pyg(viewer=k)` already exists; just plumb into dataset wrapper |

**Total ~8 hours of engineering** before Week 1's training run, all
small. None requires architectural changes.

---

## 7. Success criteria

By end of Phase 4:

- [ ] GnnMcts winrate vs LookaheadMcts in 4-way tournament: **≥25%**
  (= uniform baseline; meaning the GNN-MCTS is at least as good as
  Random vs LookaheadMcts).
- [ ] PureGnn winrate vs Random in 1v1: **≥50%** (clearly above
  coin-flip).
- [ ] At least one trained GNN that beats the original Phase 3.5
  baseline checkpoint in head-to-head.
- [ ] Self-play loop runs cleanly to ≥5 iterations without divergence.
- [ ] Full pipeline (engine + recorder + train + tournament) still
  passes safety harness.

If we hit all five, we've shipped a real Catan-playing GNN within our
hardware budget. If we hit only the first three, we've shipped a
weaker-than-LookaheadMcts but useful-as-evaluator GNN. Either is a
real result — better than where Phase 3 ended.

---

## 8. Sources

- [Catanatron repo](https://github.com/bcollazo/catanatron)
- [Bryan Collazo — "5 Ways NOT to Build a Catan AI" (Medium)](https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17)
- [Henry Charlesworth — Learning To Play Settlers of Catan With Deep RL](https://settlers-rl.github.io/)
- [Charlesworth's settlers_of_catan_RL GitHub](https://github.com/henrycharlesworth/settlers_of_catan_RL)
- [QSettlers (Krishna)](https://akrishna77.github.io/QSettlers/)
- [Re-L Catan (Stanford CS230)](http://cs230.stanford.edu/projects_fall_2021/reports/103176936.pdf) — PDF unparseable
- [Justin Asher — Modeling Catan through self-play](https://justinasher.me/catan_ai)
- [Playing Catan with Cross-dimensional NN — arxiv.org](https://arxiv.org/pdf/2008.07079)

---

## 9. Phase 3 closeout

Phase 3 ended with:
- Engine: feature-complete, well-tested, fast, salvage-safe.
- Pipeline: working end-to-end on GPU, validated by tournament.
- GNN: trained, weak (val_top1=0.273, 0% in tournament).
- Honest negative result documented, with concrete next moves.

Phase 4 plan above is the next iteration. **None of the Phase 4 work
requires re-doing any Phase 3 work** — the engine, sweep, baseline
GNN, and tournament harness all stay. Phase 4 adds three new
capabilities (simplified rules, imitation learning, self-play loop)
that build on what we have.
