# Why more data didn't lift PureGnn — root-cause diagnosis

**Date:** 2026-06-01
**Trigger:** ckpt5 confirmed PureGnn vs LookV3 plateaus at ~6% even with 509
games / 305k positions (trajectory 8.3->11.7->1.7->6.0, all noise). User: "we
are NOT shipping GnnMcts — diagnose why it didn't improve with data."

## D1 — Optimization: NOT the problem (the net trains fine)

Policy-loss curves across data scales (from training_log.json):
| corpus | train_pol ep1->last | val_pol ep1->last |
|---|---|---|
| ckpt1 110g | 2.083 -> 1.940 | 2.094 -> 2.130 |
| ckpt3 323g | 1.996 -> 1.916 | 2.084 -> 2.159 |
| ckpt5 509g | 2.008 -> 1.957 | 2.060 -> 2.108 |

- The VALUE head fits beautifully (train_val 0.21 -> 0.009) — so the net CAN
  learn; optimization works.
- But the POLICY barely moves (train_pol ~2.0 -> ~1.95) and val_pol RISES from
  ep1 (overfits immediately). The policy is essentially unlearnable here.
- **More data makes ZERO difference to policy loss** (ckpt1 110g == ckpt5 509g,
  both ~2.0 train / ~2.1 val). This rules out "not enough data" AND "optimization
  stuck" — it's a TARGET or CAPACITY wall.

## D3-prelim — Target quality: THE dominant cause

Inspected the visit-count POLICY TARGETS on 3000 sampled states from the
509-game corpus:
- **mean target entropy = 1.473 nats** (the CE floor if the net matched targets)
- net's actual policy CE ~1.95 -> it's BETWEEN the floor (1.47) and uniform (2.63)
- **mean peak visit-share = 0.53** (1.0 = decisive search, ~0.07 = flat)
- **30% of targets are NEAR-FLAT (peak < 0.3)** — the sims=160 search spread its
  visits roughly evenly, expressing NO clear best move
- only 36% are decisive (peak > 0.7)

**Interpretation:** at sims=160 the MCTS frequently fails to concentrate on a
best move (30% of states it's effectively undecided). So a large fraction of the
policy TARGETS are blurry "these moves are ~equal" distributions. The net learns
a blurry policy -> **argmax of a blurry policy ≈ random among the top moves** ->
PureGnn plays weakly. More blurry-target DATA can't sharpen what the search
itself didn't resolve. This is why data volume didn't help: the LABELS, not the
quantity, are the bottleneck.

## Hypotheses status
- H1 (target quality): STRONGLY SUPPORTED — 30% flat targets, policy CE far from
  the soft-target floor, flat across data.
- H2 (capacity h128): still possible (val_top1 flat ~0.37) — TEST: train h256.
- H3 (optimization): RULED OUT — value head fits fine, net trains.
- H1b (exploration poison): plausible contributor — temperature sampling +
  Dirichlet noise add variance to which moves get explored. TEST: greedy targets.

## Next diagnostics
- D2: train h256 on the 509-corpus — does policy loss drop? (capacity)
- D3: re-search a sample at sims=800 — do targets sharpen (peak share rise)?
  (confirms sims=160 is too shallow)
- D4: train on GREEDY (no-exploration) self-play targets — sharper? (poison)

## D3 — Target sharpness sims=160 vs 800: blur is INTRINSIC, not shallow-search

Re-searched 25 corpus states at sims=160 vs sims=800 (Cell6 net):
| | peak-share | entropy | frac flat (peak<0.3) |
|---|---|---|---|
| sims=160 | 0.539 | 1.446 | 0.36 |
| sims=800 | 0.591 | 1.289 | 0.28 |

- 5x more sims sharpens the targets only MARGINALLY (+0.05 peak share). So "use
  higher sims for sharper targets" is NOT the fix — the blur is largely intrinsic.
- argmax(160) == argmax(800) only **72%** — 28% of the time deeper search picks
  a DIFFERENT best move, so sims=160 targets are also somewhat NOISY in which
  move they prefer.

**Conclusion:** many Catan positions genuinely have SEVERAL near-equal-value
moves (multi-modal optimum). The visit-count targets honestly reflect that
(flat + sometimes shifting argmax). This rules out H1a (shallow search) and H1
(fixable by better labels). It points at:
- H5 (argmax-is-wrong): a raw policy FORCED to argmax a multi-modal target picks
  poorly — exactly the regime where SEARCH (evaluates each candidate) wins and
  raw policy (must commit to one) loses. Structural to value-based games.
- H2 (capacity): a small h128 net may also lack the capacity to even represent
  the multi-modal distribution sharply.

## Refined root cause
The PureGnn plateau is NOT a data-volume or shallow-search problem. It is
STRUCTURAL: Catan decision states are frequently multi-modal (several good
moves), so (a) the policy targets are inherently soft/ambiguous, and (b) argmax
of a soft policy throws away the value information that distinguishes the
near-equal moves. Search recovers that value at decision time; the raw policy
cannot. This is why GnnMcts >> PureGnn and why more data doesn't close the gap.

## Remaining tests
- D2 (capacity h256): does a bigger net lower policy loss / raise PureGnn? If NOT
  -> confirms it's structural (argmax), not capacity.
- D4 (greedy vs exploratory targets): minor; exploration adds some noise but D3
  shows the core blur is intrinsic.

## D2 — Capacity test (h256 vs h128): NOT capacity-limited

Trained h256 (num_layers=4, ~4x params of h128) on the FULL 509-game corpus
(305,601 positions), same cache, lr 5e-4, rotate, early-stop patience 2.

| Net | best val_top1 | train_loss floor | per-game top1 |
|---|---|---|---|
| h128 (all prior cells) | ~0.37 | ~1.95 | — |
| **h256 (D2)** | **0.373** (ep1) | 1.972 (ep4) | [0.29\|0.33\|0.36\|0.39\|0.46] |

- Doubling hidden dim (4x parameters) moved val_top1 by **+0.003** — noise.
- Early-stopped at ep4; best was ep1. val_loss FLAT at 2.86 across all 4 epochs
  — the bigger net does not extract more signal from the same targets.

**Conclusion: H2 (capacity) is RULED OUT.** The h128 net is not the bottleneck;
a 4x-larger net fits the policy targets no better. The policy ceiling (~0.37
val_top1) is a property of the TARGETS, not the model size.

## FINAL ROOT-CAUSE VERDICT — why more data didn't lift PureGnn

Four diagnostics triangulate to one structural cause:
- **D1** — net trains fine; VALUE head fits, POLICY head doesn't; policy loss flat
  110g→509g. Not optimization, not data volume.
- **D3** — 5x more search sims (160→800) barely sharpens targets (+0.05 peak),
  and 28% of the time argmax shifts. The label blur is INTRINSIC to the positions.
- **D2** — 4x bigger net (h256) gives identical val_top1 (0.373 vs ~0.37). Not
  capacity.
- (target entropy) — 30% of visit-count targets are near-flat.

**THE CAUSE — multi-modal optima + argmax destroys value information.**
Catan decision states frequently have SEVERAL near-equal-value moves. The MCTS
visit-count targets honestly encode that as soft/flat distributions. A policy
head trained on soft targets learns a soft (correct!) distribution — but PureGnn
DEPLOYS via argmax, which collapses that distribution to one move and throws away
exactly the value information that distinguishes the near-equal candidates. Search
(GnnMcts) recovers it by simulating each candidate. This is why:
- GnnMcts beats LookV3 (53.8%) while PureGnn stalls at ~5%,
- more data / bigger net / more sims none of it closes the gap (the gap is in the
  argmax deployment step, not the learned policy),
- the plateau is at ~5-6% regardless: it's a structural property of value-based
  combinatorial games, not a training failure.

**Therefore:** "more data" was never going to lift PureGnn vs LookV3 on this
problem. The lift requires changing the DEPLOYMENT, not the training. Options for
a shippable non-GnnMcts (or cheap-search) player are in the next section.

## Fix options (not shipping full GnnMcts)
1. **Cheap fixed-width search at deploy** (sims=8–32 PUCT, no tree reuse) — keeps
   most of search's value-recovery at a fraction of GnnMcts cost. Likely the best
   ROI: turns the argmax into a shallow lookahead that ranks the near-equal moves.
2. **1-ply value rollout (greedy-Q over legal moves)** — for each legal action,
   apply it, evaluate child with the VALUE head (which D1 shows DOES fit), pick
   argmax-Q. No tree, one batched eval per legal set. Directly exploits the one
   head that learned well. Cheapest principled fix.
3. **Policy+value blend at argmax** — rank moves by π(a)·exp(Q-tilt) using the
   value head as a tiebreaker among near-equal-π moves. Almost free.
4. (rejected) more self-play data / bigger net — D1/D2/D3 show these don't help.

Recommend prototyping #2 (1-ply value-Q PureGnn) first: it is the minimal change
that addresses the diagnosed cause (argmax discards value), reuses the head we
proved fits, and is far cheaper than GnnMcts.
