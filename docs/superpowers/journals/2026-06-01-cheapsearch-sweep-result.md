# Cheap-search sims sweep — search gets WORSE with more sims (low-sims regime)

**Date:** 2026-06-01
**Question:** ValueQ (1-ply) tied raw argmax; the gap looked like no-search-vs-search.
Find the cheapest GnnMcts sims that beats raw-PureGnn and nears LookV3.

## Result — e10g, 120 games each, same Cell6 net, shared seeds

| sims | GnnMcts win% | RawPureGnn win% | LookV3 win% |
|---:|---:|---:|---:|
| 8  | 14.2% | 15.8% | 54.2% |
| 16 | **7.5%** | 18.8% | 55.0% |
| 32 | **3.3%** | 15.8% | 65.0% |

(0 skipped/timeout; mean_batch 11; RawPureGnn stable 15.8-18.8% across runs ->
harness is sound, the GnnMcts trend is real.)

## The surprise: MORE search = WORSE play
GnnMcts DROPS 14.2 -> 7.5 -> 3.3% as sims goes 8 -> 16 -> 32. This is the OPPOSITE
of "search recovers value." None of the cheap-search points beats raw argmax.

We KNOW GnnMcts@sims=200 hits ~54% vs LookV3 (Gate 2). So low-sims search is in a
pathological regime, NOT a flat "search doesn't help" verdict.

## Leading explanation — shallow PUCT over a wide action space
With a ~280-wide action space and only 8-32 sims, the tree barely expands past the
root's first-visited children. The visit-count argmax then reflects PUCT's
exploration bonus (c=1.4) spreading thin visits across many shallow children
rather than a real value estimate. Adding sims (16, 32) spreads visits even
thinner across breadth WITHOUT gaining depth -> the argmax-visit signal degrades
until sims is large enough to actually search to depth. So there is a U / valley:
raw argmax (good prior) > shallow search (noised prior) ... > deep search (sims=200,
real value, ~54%).

## Decisive control: sims=200 in THIS harness — CONFIRMS the valley

| sims=200 (FINAL, 120/120 games) | win% |
|---|---:|
| **GnnMcts@200** | **51.7%** |
| LookV3 | 31.7% |
| RawPureGnn | 8.3% |

GnnMcts@200 recovers to 51.7% (matches Gate-2's ~54%), beating LookV3, in the SAME
harness that gave the low-sims numbers. The harness is sound; RawPureGnn stays low
(6.1%) as expected. **The low-sims valley is REAL, not a bug.**

Full sims curve (this harness, same net + seeds):
| sims | 0(argmax) | 1-ply Q | 8 | 16 | 32 | 200 |
|---|---|---|---|---|---|---|
| GnnMcts win% | ~13-18 | ~14 | 14.2 | 7.5 | 3.3 | **51.7** |

A clear VALLEY: prior alone is OK, shallow search NOISES it (worse, bottoming
~sims=32 at 3.3%), and only deep search (~200) crosses into real value recovery.

## Implication — the valley is confirmed
- "Search-free or cheap-search deploy beats LookV3" looks FALSE on this stack:
  1-ply ties argmax; 8-32 sims is worse than argmax; only ~200 sims works.
- That re-centers the deployable on the known winner (GnnMcts@~200, 54%) OR on
  improving the RAW policy/value so argmax itself is stronger (back to training,
  but D1/D2/D3 say that's capped at this scale).
- Worth testing the PUCT hypothesis directly: lower c_puct at low sims, or
  policy-target temperature, might rescue cheap search. Defer pending control.

## VERDICT (control confirmed)
On this stack, the ONLY deployment that beats LookV3 is real search at ~200 sims —
which is exactly "GnnMcts." Every cheaper deploy failed: 1-ply value-Q ties argmax;
8-32-sim search is WORSE than argmax. The diagnosed plateau cause (multi-modal
optima + argmax discards value) is correct, but the fix is NOT cheap — it needs
enough search depth to overcome the wide-action-space PUCT noise floor (~tens of
sims wasted before depth helps).

This re-centers the deployable decision (to discuss with user, given "not shipping
gnn+mcts"):
1. Accept GnnMcts@~150-200 as the deployable (it's the only ~50%+ player). Cost is
   the open question — measure ms/move; it may be acceptable for non-realtime use.
2. Try to RESCUE cheap search by killing the PUCT noise: lower c_puct (e.g. 0.5),
   FPU/prior-trust tweaks, or fewer-but-wider root children. Could move the valley
   floor up. Cheap experiment, high information.
3. Improve the RAW policy/value so argmax itself is stronger — but D1/D2/D3 say
   this is capped at h128 + few-hundred-game scale; would need a different training
   signal (e.g. value-weighted policy targets), not just more data.
Recommend (2) as the next cheap probe (it directly tests the valley mechanism),
then bring the cost numbers for (1) so the user can decide what "shippable" means.
