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

## Decisive control (running): sims=200 in THIS harness
If sims=200 recovers ~50%+ here, the low-sims numbers are confirmed as the
shallow-PUCT valley -> conclusion: cheap search does NOT work; you need real sims
(expensive). If sims=200 is ALSO low here, there is a harness/search bug to fix
before trusting any of this. Either way we learn something decisive.

## Implication if the control confirms the valley
- "Search-free or cheap-search deploy beats LookV3" looks FALSE on this stack:
  1-ply ties argmax; 8-32 sims is worse than argmax; only ~200 sims works.
- That re-centers the deployable on the known winner (GnnMcts@~200, 54%) OR on
  improving the RAW policy/value so argmax itself is stronger (back to training,
  but D1/D2/D3 say that's capped at this scale).
- Worth testing the PUCT hypothesis directly: lower c_puct at low sims, or
  policy-target temperature, might rescue cheap search. Defer pending control.
