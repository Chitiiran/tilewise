# Catan Bot (`tilewise`) — Project Synthesis & Handoff

**Compiled:** 2026-06-10
**Resume point of record:** `docs/superpowers/journals/2026-06-01-puregnn-deploy-investigation-FINAL.md`
**Purpose:** Summarize the objective, the attempts so far, and exactly where the project is stuck, with critical insight a superior model can use to finish the task. The handoff prompt is in §6.

> All file/result claims below were verified against disk on 2026-06-10 (cited journals, specs, and infra `.py` files all exist; the June-1 FINAL journal is the explicit "resume point"). Numbers are quoted from the journals, not from memory.

---

## 1. The objective

**Build a graph neural network that plays full Catan well — specifically, a GNN policy/value net that beats the existing hand-coded `LookaheadMctsV3` bot, ideally as a cheap policy-only player ("PureGnn", argmax, no search at deploy time).**

The substrate is `tilewise`: a headless, deterministic Rust Catan engine (PyO3-bound) built for million-games/sec self-play, with a graph-native board (hex/vertex/edge heterograph) chosen so a GNN is the natural model. The long-term destination is **AlphaZero-style self-play training** on full Catan (10 VP, longest-road + largest-army bonuses on).

---

## 2. The arc of attempts (chronological, compressed)

| Phase | What was tried | Outcome |
|---|---|---|
| **v1 engine + MCTS study** | Rust engine, OpenSpiel `MCTSBot` adapter, chance-node API, 4-experiment study, bootstrap self-play parquet | ✅ Done. Training data + reusable learnings (chance nodes non-negotiable; random-rollout game length is empirical ~12–15k steps, not the spec's ~80). |
| **GNN v0 + v2 engine** | First GNN trained on MCTS-vs-random data; engine rebuilt to full Catan rules (v2) | GNN v0 lost 0/4 seats vs LookaheadMcts. Honest negative. |
| **v3 "Catan-Lite"** | Simplify win condition (5 VP, bonuses off) via engine flags to train a stronger GNN *faster*, then port back. Generated 100k-game self-play corpus. | Trained but didn't beat LookV3. Scaffolding only — always meant to port back to full Catan. |
| **Loss-augmentation roadmap** | Auxiliary loss terms (Cand 1/2/7/8/10) on top of supervised CE | Mostly negative. Established methodology + the rule that `val_top1` misleads under loss-aug. |
| **Cand 11 + cumulative-best inversion** | Cand 11 = pure road-pip prior. 8 cells trained, four 1200-game tournaments | **Key discovery: the "best" cell is rule-conditional.** Cell 5 v2 (Cand 11 alone) wins v3 rules; **Cell 6 (Cand 11+8+10 stack) wins full Catan** (54.33% no-LookV3, 19.00% with LookV3). The "dev-card-spam degenerate equilibrium" they'd worried about is actually the **load-bearing Largest-Army strategy** for full Catan. |
| **GNN+MCTS question (e10e)** | Does bolting MCTS onto the GNN help? | First measurement said **GNN+MCTS worse than PureGnn** (1 vs 8 wins) — but confounded by a value-perspective bug + 32% GPU-contention timeouts. |
| **Value-perspective bug fix + batched eval** | GNN value head is ego-relative but MCTS indexed it as absolute seat → poisoned Q-values. Fixed in async MCTS. Built `BatchedGnnEvaluator` (4.3× speedup, 256→59 s/game). | After the fix, **GnnMcts@200 ≈ 51–54% vs LookV3** — it *does* beat LookV3. The earlier "MCTS hurts" finding was largely the bug + contention. |
| **AlphaZero exploration (May 31)** | Dirichlet noise + temperature sampling self-play, warm-started from Cell 6, arena-gated, 9 h compounding loop (323 games) | **Negative on winrate, positive mechanistically.** Policy demonstrably learned the trade behavior it was missing (override rates dropped), but PureGnn stayed ~6% vs LookV3 ~45%. AZ needs *millions* of games; at few-hundred-game scale variance swamps signal. Code correct & committed; blocker is throughput, not the algorithm. |
| **PureGnn deploy valley (June 1, latest)** | Can we ship a *cheap* (no/shallow-search) net that beats LookV3? Four diagnostics (D1/D2/D3 + target entropy) + multiple deploy variants. | See §3 — this is where it's stuck. |

---

## 3. Where it is stuck (current frontier, 2026-06-01)

**Core unsolved problem: PureGnn (raw argmax policy, no search) plateaus far below LookV3, and this is structural — more data and bigger nets do not fix it.**

The June-1 investigation nailed the mechanism conclusively (four committed diagnostics):

| Test | Rules out | Finding |
|---|---|---|
| **D1** optimization | "net isn't fitting" | VALUE head fits; POLICY head doesn't; policy loss flat across 110→509 games |
| **D3** target sharpness | "sims too shallow → blurry labels" | 5× sims (160→800) barely sharpens (+0.05 peak); 28% argmax shifts → blur is **intrinsic** |
| **D2** capacity (h256) | "h128 too small" | `val_top1` 0.373 ≈ h128's ~0.37 — 4× params, no gain. **Not capacity** |
| target entropy | — | ~30% of visit-count targets near-flat |

**Root cause (structural):** Catan decision states frequently have *several near-equal-value moves*. MCTS visit-count targets honestly encode that as soft/flat distributions. The policy head learns the soft (correct) distribution — but **PureGnn deploys via argmax, which collapses it to one move and discards the value information that distinguishes the near-equal candidates.** Search recovers that value at decision time; raw argmax cannot. So "more data" was never going to fix PureGnn — the gap is in the **deployment (argmax)**, not the learned policy.

**Every cheap deploy fix failed, revealing a "valley"** (same Cell 6 net, shared seeds, 120 games each):

| Deploy method | Winrate vs field | Verdict |
|---|---:|---|
| Raw argmax (PureGnn) | ~8–18% | baseline |
| 1-ply value-Q tiebreak (`ValueQGnnBot`) | ~14% | **ties argmax** — fails |
| Cheap search sims=8 | 14.2% | worse-ish |
| Cheap search sims=16 | 7.5% | WORSE than argmax |
| Cheap search sims=32 | 3.3% | WORST |
| **Real search sims=200 (GnnMcts)** | **51.7%** | beats LookV3 (31.7%) — the only winner |

**The valley:** good prior (argmax) > shallow search (noises the prior) > deep search (real value). Shallow PUCT over the ~280-wide action space spreads thin visits across breadth without depth, degrading the argmax-visit signal. The sims=200 control (51.7%) reproduces Gate-2 (~54%) *in this harness* while RawPureGnn stays low → **the harness is sound and the valley is real, not a bug.**

**Why 1-ply specifically failed:** it evaluates mid-turn, off-distribution children (where the value head is least calibrated) and is blind to the opponent's reply. Catan's near-equal moves differ in what they *enable next turn* — invisible to 1 ply.

**The conclusion the data forces:** on this stack, the **only** deployment that beats LookV3 is real ~200-sim search — which *is* "GnnMcts". This conflicts with the standing "not shipping GNN+MCTS" goal (it's slow: ~4 min/game single-worker).

---

## 4. The decided-but-unbuilt next step — distillation

The user chose this before saying "document everything and stop":

- **Approach — policy distillation.** Generate self-play where `GnnMcts@200` (the 51.7% teacher) selects moves; train PureGnn's **policy** to imitate the **teacher's decision** (not the raw visit counts). The raw argmax then inherits the searcher's value-discrimination → strong play with **no search at deploy time**.
- **Target form — NOT finalized (the open question we stopped on).** Options on the table:
  1. **Sharpened visit-count distribution (visits² renormalized) — recommended in the journal.**
  2. Hard argmax label (the move the teacher played).
  3. Sharpened visits + keep the value-head auxiliary (multi-task).
- **Why it's the one untried lever:** D1/D2/D3 ruled out more-data and bigger-net *at fixed targets*. They did **not** test a *better target*. Distillation changes the target to encode what deep search concluded — the one thing the soft visit-count target fails to capture.
- **Cost note (unmeasured):** one sims=200 teacher search per recorded state is expensive; the teacher data-gen run is the main cost driver. Size it first from the e10g control throughput (120 games used `total_batches=136,813` — measure states/sec there).

---

## 5. Critical insights for whoever finishes this

Non-obvious things that took the project months to learn:

1. **Train and evaluate in the target distribution.** The biggest methodological lesson. Models that look weak in v3 (simplified rules) carry the load-bearing strategy for full Catan; v3 *actively biased* some cells against full-Catan competence (Cand 11's road prior teaches "expand fast to 5 VP", wrong for 10-VP closeout). **Generate any new self-play data with `bonuses=True, vp_target=10`.**

2. **Metrics lie, ranked by how much:**
   - `val_top1` is NOT a winrate proxy, *especially* under loss-augmentation.
   - **Mid-tournament metrics systematically understate strong cells (~6 pp).** Canonical measurement is `e10c_triple_gnn` @ 1200 games. The Cell 5 v2 vs Cell 6 ranking flipped *three times* depending on the measurement.
   - **Wall-clock timeouts silently inflate winrates** (the slow seat's losses go uncounted). Demand **<5% timeout rate** or the number is an artifact (the recurring "e5 wall-clock artefact" lesson).

3. **The value-perspective bug is the key un-confounder.** The GNN value head is **ego-relative** (`value[0]` = current mover), but the old `gnn_evaluator.py` + OpenSpiel MCTS indexed it as **absolute seat**, poisoning Q-values for non-mover nodes — the likely cause of the original "GNN+MCTS is worse" finding. Fixed in the *async* MCTS; **verify any value-routing code rotates ego→absolute before trusting it.**

4. **The deploy valley is real, not a harness bug** (sims=200 control reproduces ~54% in-harness; RawPureGnn stays low). Don't re-litigate whether shallow search "should" help — it measurably doesn't here.

5. **The untested lever is a better *target*, not more data.** D1/D2/D3 ruled out more-data and bigger-nets at fixed targets. Distillation (§4) is the genuinely open thread; **the real open design decision is the distillation target form** — start there, don't re-run diagnostics.

6. **AlphaZero isn't disproven — it's throughput-blocked.** The AZ code is correct and committed and demonstrably moves the policy the right direction. It needs millions of games; the box (single 4 GB GTX 1650, CPU-bound at ~0.8 games/min/worker) can't produce them. `BatchedGnnEvaluator` (built) is the enabling infra; further speedups are **CPU-side** (multiprocessing, vectorize `state_to_pyg`), not GPU.

7. **Two viable end-states exist today.** (a) Ship **GnnMcts@200** — works (51.7%) but violates the "no search at deploy" preference and is slow. (b) **Distill it into a cheap PureGnn** — the unbuilt bet for a fast, search-free 50%+ player.

---

## 6. Handoff prompt (copy-paste for a superior model)

> **Context:** You are taking over `tilewise`, a Rust+Python Catan engine + GNN project whose goal is a neural net that beats the hand-coded `LookaheadMctsV3` bot at **full Catan** (10 VP, longest-road + largest-army bonuses on), ideally as a **cheap policy-only player (PureGnn, argmax, no search)**.
>
> **State of play (2026-06-01):** The strongest model is **Cell 6** (GNN trained with stacked loss-aug: Cand 11 road-pip prior + Cand 8 BuyDevCard prior + Cand 10). With ~200-sim MCTS on top (`GnnMcts@200`, after fixing a value-perspective bug) it **beats LookV3 at ~51.7%**. But the project wants to *avoid shipping search*, and **raw PureGnn plateaus at ~8–18%** vs LookV3.
>
> **The conclusively-diagnosed blocker:** PureGnn's plateau is *structural*, not a data/capacity problem (proven: policy loss flat with more data; h256≈h128; 5× sims barely sharpens targets). Catan has many near-equal-value moves → visit-count targets are honestly soft → **argmax discards the value info that breaks those ties.** Search recovers it; argmax can't. Cheap search makes it *worse* (a real "valley": sims=8/16/32 → 14.2/7.5/3.3%) because thin PUCT visits over the ~280-wide action space spread across breadth without depth.
>
> **Your task — pick up the one untested lever: distillation.** All prior diagnostics ruled out more-data and bigger-nets *at a fixed training target*. They did NOT test a *better target*. The decided-but-unbuilt next step is to **distill `GnnMcts@200` (the 51.7% teacher) into the raw PureGnn policy** so argmax inherits search's value-discrimination. **The first real decision (un-made, yours to make) is the distillation target form** — candidates: (a) **sharpened visit counts (visits² renormalized) — the journal's recommendation**, (b) hard argmax of the teacher's chosen move, (c) sharpened visits + an explicit value-auxiliary loss. Reason about which best transfers the tie-breaking value signal, then implement and measure. Before launching the teacher data-gen run, size it from the e10g control throughput (120 games used `total_batches=136,813` → states/sec).
>
> **Hard rules (learned the hard way — violate these and you'll be fooled):**
> 1. Generate any new self-play data at **full Catan rules** (`bonuses=True, vp_target=10`). v3-simplified rules actively bias the policy wrong.
> 2. **Canonical measurement is `e10c_triple_gnn` @ 1200 games.** Mid-tournament metrics understate strong cells ~6 pp; `val_top1` is NOT a winrate proxy. Demand **<5% timeout rate** or the winrate is a wall-clock artifact.
> 3. The GNN value head is **ego-relative**; the old `gnn_evaluator.py` indexed it as absolute seat (a bug). Verify any value-routing code rotates ego→absolute before trusting it.
> 4. Environment: ML runs in **WSL** (OpenSpiel has no Windows wheels); `runs/v3/` is a WSL symlink off C:; run **`maturin develop --release`** after any Rust change.
>
> **Start by reading:** `docs/superpowers/journals/2026-06-01-puregnn-deploy-investigation-FINAL.md` (the resume point / full diagnosis), its companions `2026-06-01-puregnn-plateau-diagnosis.md`, `2026-06-01-cheapsearch-sweep-result.md`, `2026-06-01-valueq-gate-result.md`, then `docs/superpowers/specs/2026-05-30-batched-gnn-evaluator-design.md` (infra you'll lean on).
>
> **Reusable infra already committed:** `catan_mcts/value_q_bot.py`, `catan_mcts/experiments/e10f_valueq_async.py`, `catan_mcts/experiments/e10g_cheapsearch_async.py`, `analyses/score_e10f.py`, `analyses/score_e10g.py`, `catan_mcts/async_mcts.py` (with Dirichlet/temperature exploration), `BatchedGnnEvaluator`. Teacher net: `runs/v3/rl_checkpoints/round0_Cell6.pt` (h128, L4). Training stack (`catan_gnn/train.py`) already supports auxiliary policy-loss terms — a distillation loss slots in the same way.
>
> **Fallback if distillation fails:** the honest, already-working deliverable is **GnnMcts@200 at full Catan (~51.7% vs LookV3)**. AlphaZero self-play is *correct and committed* but throughput-blocked (needs millions of games; box is CPU-bound at ~0.8 games/min/worker) — pursue it only after first solving self-play throughput (next levers are CPU-side: multiprocessing, vectorizing `state_to_pyg`).

---

## 7. Environment caveats (will bite immediately)

- **OpenSpiel has no Windows wheels.** Python ML work runs in **WSL Ubuntu** (venv in the Linux fs). Windows side handles git.
- **`runs/v3/` is a WSL symlink** to `/home/chitii/catan_data/runs/v3/` (off the C: drive). Windows tools see a broken symlink — use UNC path `\\wsl.localhost\Ubuntu\home\chitii\catan_data\runs\v3\`.
- **Rebuild PyO3 after any Rust change:** `maturin develop --release` (pytest is a false negative otherwise). After a worktree swap, the editable install breaks → rebuild from the new worktree.
- The full-history `v3` branch holds the loss-aug work; `main` has only part of it. Active worktrees live under `.claude/worktrees/` (e.g. `v4/`, `interactive-play/`) — the canonical infra copies are at the repo root `mcts_study/`.
