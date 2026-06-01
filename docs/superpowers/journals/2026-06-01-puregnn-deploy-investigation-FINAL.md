# PureGnn vs LookV3 — full diagnosis + deploy-fix investigation (FINAL, 2026-06-01)

Consolidated report for the question: **why doesn't more data make PureGnn (raw
GNN policy, no search) beat LookV3, and can we fix it without shipping GnnMcts?**
User decision at the end of this session: **document everything and stop** (do not
build the distillation fix yet). This file is the resume point.

---

## 1. The diagnosis — why more data didn't lift PureGnn (CONCLUSIVE)

Four diagnostics, all committed (see `2026-06-01-puregnn-plateau-diagnosis.md`):

| test | rules out | finding |
|---|---|---|
| **D1** optimization | "net isn't fitting" | VALUE head fits; POLICY head doesn't; policy loss FLAT across 110->509 games |
| **D3** target sharpness | "sims too shallow -> blurry labels" | 5x sims (160->800) barely sharpens (+0.05 peak); 28% argmax shifts -> blur is INTRINSIC |
| **D2** capacity (h256) | "h128 too small" | val_top1 0.373 ≈ h128's ~0.37 — 4x params, no gain. NOT capacity |
| target entropy | — | 30% of visit-count targets near-flat |

**Root cause (structural):** Catan decision states frequently have SEVERAL
near-equal-value moves. The MCTS visit-count targets honestly encode that as
soft/flat distributions. The policy head learns a soft (correct) distribution —
but PureGnn DEPLOYS via argmax, which collapses it to one move and discards the
value information that distinguishes the near-equal candidates. Search recovers
that value at decision time; raw argmax cannot. => "more data" was never going to
fix PureGnn; the gap is in the DEPLOYMENT (argmax), not the learned policy.

## 2. Deploy-side fixes tried — ALL FAILED (this session's experiments)

Tested every way to beat LookV3 WITHOUT expensive search. Same Cell6 net, shared
seeds, e10f/e10g async harnesses, 120 games each. (Win% is field-dependent across
harnesses; the WITHIN-run comparisons below are apples-to-apples.)

| deployment | win% vs field | verdict |
|---|---:|---|
| raw argmax (PureGnn) | ~8-18 | baseline |
| **1-ply value-Q** (ValueQGnnBot) | ~14 | **TIES argmax** — fails (`2026-06-01-valueq-gate-result.md`) |
| cheap search sims=8 | 14.2 | worse-ish |
| cheap search sims=16 | 7.5 | WORSE than argmax |
| cheap search sims=32 | 3.3 | WORST |
| **real search sims=200** | **51.7** | beats LookV3 (31.7) — the only winner |

**The valley:** good prior (argmax) > shallow search (NOISES the prior) > deep
search (~200 sims, real value). Shallow PUCT over a ~280-wide action space spreads
thin visits across breadth without depth, degrading the argmax-visit signal. The
sims=200 control (120 games, 51.7%) reproduces Gate-2 (~54%) IN THIS HARNESS and
RawPureGnn stays low throughout -> harness is sound, the valley is REAL not a bug.
(`2026-06-01-cheapsearch-sweep-result.md`.)

**Why 1-ply failed specifically:** it evaluates mid-turn, off-distribution
children (value head least calibrated there) and is blind to the opponent's reply.
Catan's near-equal moves differ in what they ENABLE next turn — invisible to 1 ply.

## 3. The conclusion the data forces

On this stack, **the ONLY deployment that beats LookV3 is real ~200-sim search —
which IS "GnnMcts."** Every cheaper deploy (argmax, 1-ply value-Q, 8/16/32-sim
search) is at or below raw argmax. The diagnosed cause is correct but the fix is
not cheap: it needs enough search depth to clear the wide-action-space PUCT noise
floor. This directly conflicts with the standing "not shipping gnn+mcts" goal.

## 4. Decided next step (NOT yet built) — distill GnnMcts@200 into the raw policy

User picked, before saying stop:
- **Approach:** policy distillation. Generate self-play where GnnMcts@200 (the
  51.7% teacher) selects moves; train PureGnn's POLICY to imitate the TEACHER's
  decision, not the raw visit counts. The raw argmax then inherits the searcher's
  value-discrimination => strong play with NO search at deploy time.
- **Target form:** NOT FINALIZED (this is the open question we stopped on). The
  options on the table were:
  1. Sharpened visit-count distribution (visits^2 renormalized) — recommended.
  2. Hard argmax label (the move the teacher played).
  3. Sharpened visits + keep value-head auxiliary (multi-task).
- **Why it's the one untried lever:** D1/D2/D3 ruled out more-data and bigger-net
  at FIXED targets. They did NOT test a BETTER target. Distillation changes the
  target to encode what deep search concluded — the one thing the soft visit-count
  target fails to capture.
- **Cost note (unmeasured):** one sims=200 teacher search per recorded state is
  expensive; a teacher data-gen run is the main cost driver. Estimate before
  launching (cite throughput from the e10g sims=200 run: 120 games used
  total_batches=136,813 -> measure states/sec there to size the corpus run).

## 5. Infra built this session (committed, reusable)
- `catan_mcts/value_q_bot.py` — ValueQGnnBot (1-ply value-Q) + chance-node
  expected-value handling. 6 tests in `tests/test_value_q_bot.py`. (Result: ties
  argmax — kept for the record and as a value-head probe.)
- `catan_mcts/experiments/e10f_valueq_async.py` — ValueQ vs raw-PureGnn vs LookV3.
- `catan_mcts/experiments/e10g_cheapsearch_async.py` — GnnMcts@--sims vs raw vs
  LookV3 (the sims sweep + control harness).
- `analyses/score_e10f.py`, `analyses/score_e10g.py` — rotation-aware role scorers.
- `run_e10g_sweep.sh` — sequential sims sweep driver.
- Training stack (`catan_gnn/train.py`) already supports auxiliary policy-loss
  terms (Cand 7 class-balance, Cand 8 VP-prior KL) — a distillation loss slots in
  the same way.

## 6. Run artifacts (WSL, off-C:)
- `runs/v3/e10f_gate/` — ValueQ gate (120g): ValueQ 14.2, RawPureGnn 13.3, LookV3 59.2
- `runs/v3/e10g_sweep_sims{8,16,32}/` — cheap-search sweep (120g each)
- `runs/v3/e10g_control_sims200/` — control (120g): GnnMcts 51.7, LookV3 31.7, Raw 8.3
- Teacher net for distillation: `runs/v3/rl_checkpoints/round0_Cell6.pt` (h128, L4)

## 7. To resume
Pick the distillation target form (sec 4), then: brainstorm -> spec -> plan ->
build. First measurement to size the run: teacher data-gen throughput (states/sec
at sims=200) from the e10g control. The deployable-decision fork (rescue cheap
search via lower c_puct / accept reduced-sims GnnMcts / distill) is documented in
`2026-06-01-cheapsearch-sweep-result.md` sec "VERDICT".
