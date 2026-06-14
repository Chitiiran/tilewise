# Candidate Self-Play AZ Redesign — fix the stale-data root cause

**Date:** 2026-06-14
**Status:** approved (user, 2026-06-14 — all decisions locked + consistency-checked)
**Branch:** `az-difficulty-bots` (PR #3)
**Fixes:** the stale-data RCA (`2026-06-14-RCA-fresh-ratio-failed-to-prevent-stale-data.md`)
**Supersedes:** the fresh-ratio mechanism in `2026-06-13-faithful-az-daily-runner-design.md` §5

## 1. Problem being fixed

The fresh-ratio design counted data as "fresh" by **champion NAME**, so while the
champion was sticky (HOLDs), old champion-tagged games counted as fresh forever →
deficit 0 → **iterations generated 0 new games and retrained on a fixed corpus**
(iters 4–5 trained on identical iter-3 dirs). This redesign makes data **always
come from the latest network**, the way canonical AlphaZero works (self-play uses
the newest net; the arena only gates promotion).

## 2. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Games per iteration | **1000** | enough signal/iter; ~7.4h self-play |
| Self-play generator | **latest candidate** (iter-1 → champion fallback) | always-fresh data; breaks the sticky-champion stalemate |
| Window | **dynamic, accumulating** (champion + every candidate's games) | champion games anchor; bad candidates progressively diluted |
| Window reset | **on promotion** | a promotion means prior data is now weaker-policy; restart the "10 iters to improve" clock |
| Bad-candidate games | **keep all (no gating)** | dynamic window's anchor + dilution handle them; arena still gates *promotion* |
| Promote bar | **65%** (strictly greater) | demand clear improvement, not coin-flip |
| Arena size | **300 games** | 65% measurable: CI ±6.6pp (vs ±11pp at 120) — a true 65% clears reliably |
| Max iterations | **10 per model**, then STOP + user decides | bounded; stagnation guard is the softer interim stop |
| Terminology | **generator vs champion**, `gen_iter` tag | the naming conflation WAS the bug; make it impossible to repeat |

**Cost (measured/extrapolated):** ~13.9h/iter (7.4h self-play + 6.2h arena +
~0.25h train); ~5.8 days for a 10-iteration run. RAM: ≤10k-game window ≈ 11 GB,
well under the ~27k-game / 30 GB replay-path ceiling. Accepted by user
(observability + pause cover it).

## 3. Terminology & file-naming fix (loop 5 — prevents the recurrence)

| Confusing now | Fixed to |
|---|---|
| `champion` = both the self-play net AND the gate-winner | **`champion`** = crowned gate-winner only. **`generator`** = the net that produced a batch of self-play data (champion or a candidate). |
| `meta.json: {champion, rules_id}` | **`meta.json: {generator_name, gen_iter, rules_id}`** — records WHICH net made the games and WHICH iteration. The missing dimension that caused the bug. |
| `fresh_deficit(champion=...)` counts by name | **counts by `gen_iter` recency**, never by name. |
| dir name `...-self_play_async-pNNN` | **`...-gen_<name>_iter<N>-pNNN`** — a glance at the dir tells you who made it and when. |

## 4. Mechanism (replaces fresh-ratio)

Per iteration N:

1. **Choose the generator net:**
   - N == 1 (or just after a promotion-reset with no candidate yet): **champion**.
   - else: the **latest candidate** checkpoint (iter N−1's trained net), promoted or not.
2. **Generate exactly `games_per_iter` (1000) new games** with the generator,
   tagged `meta.json {generator_name, gen_iter: N, rules_id}`, into
   `iter_N/selfplay/...-gen_<name>_iter<N>-...`. Resumable: count this
   iteration's own already-done games, generate only the remainder (so a kill
   resumes toward 1000, never regenerates — and never counts OTHER iters' games
   as satisfying this iter's quota; that was the bug).
3. **Build the dynamic window:** all self-play dirs whose `gen_iter` is
   **≥ last_promotion_iter** (reset-on-promotion boundary) AND matching
   `rules_id`. (So after a promotion at iter K, only games from iters ≥ K count.)
4. **Train** the candidate (warm-start from the current champion) on the window.
5. **Arena:** candidate vs champion, **300 games**, 4 rotations × 75 shared
   seeds, VP-margin tiebreak. Promote iff candidate winrate over decisive games
   **> 0.65** (and draw-rate / min-decisive validity guards as today).
6. **Publish:** on promote → new champion + **record `last_promotion_iter = N`
   (window reset boundary)**; on hold → champion stays. PROGRESS.md row either way
   (now `generator`, `new_games`, `window_iters`, verdict, winrate, draws).
7. **Terminal:** after 10 iterations since the last promotion (or since start),
   **STOP** with a clear status flag; user decides next.

## 5. Config (AzConfig changes)

```
games_per_iter: int = 1000          # was 400; now ALWAYS generated (no skip)
arena_games:    int = 300           # was 120; 65% bar needs the statistical power
promote_threshold: float = 0.65     # was 0.55
max_iters_per_model: int = 10       # stop after 10 iters without promotion -> user
# REMOVED/retired: fresh_ratio, window_games (replaced by dynamic gen_iter window)
# self-play generator is the latest candidate, tracked via the ladder/manifest.
```

`arena_min_decisive` stays (300 games >> 40). `arena_max_draw_rate` unchanged.

## 6. State the loop must track (new)

- **`last_promotion_iter`** (in ladder.json or a loop-state file): the window
  reset boundary. Window = games with `gen_iter >= last_promotion_iter`.
- **latest candidate checkpoint path** (the generator for the next iteration):
  derivable from `iter_{N-1}/training/checkpoint_best.pt`, but recorded
  explicitly in the manifest for resumability.

## 7. Loopholes considered & resolved

- **Chicken-and-egg (no candidate on iter 1):** champion fallback. ✓
- **Bad candidate poisons window:** dynamic accumulate + champion anchor +
  progressive dilution; reset-on-promotion bounds it; arena gates promotion. ✓
- **65% unmeasurable at 120 games:** enlarged arena to 300 (CI ±6.6pp). ✓
- **Arena still uses champion (not candidate):** correct by design (self-play net
  ≠ gate opponent); explicitly separated by the generator/champion terminology. ✓
- **Compute ~doubles:** accepted (~14h/iter, ~6 days/10-iters). ✓
- **Divergence filter still drops ~0.05%/build:** unchanged, addressed later. ✓
- **Quota-satisfied-by-other-iters' games (the original bug):** killed — quota
  counts only THIS iteration's `gen_iter == N` games. ✓

## 8. Testing (TDD)

- `gen_iter`-based deficit: an iteration counts only its OWN gen_iter games toward
  its 1000 quota; **same-generator across iterations still generates new games**
  (the regression test the original bug lacked).
- generator selection: iter-1 → champion; iter≥2 → latest candidate; just-after-
  promotion-with-no-new-candidate → champion.
- dynamic window: includes only `gen_iter >= last_promotion_iter`, same rules_id.
- window reset on promotion: promote at iter K → next window excludes < K.
- promote bar 65% strictly-greater; arena 300 divisible by 4.
- terminal: 10 iters without promotion → STOP flag set.
- PROGRESS.md row carries generator + new_games + window_iters.
- integration: a micro multi-iteration run (tiny games/sims) showing the window
  growing then resetting on a forced promotion.

## 9. Migration / compatibility

- Old `meta.json {champion, rules_id}` dirs (iter-1..N already on disk) lack
  `gen_iter`. The window builder treats a missing `gen_iter` as "legacy / iter 0"
  so they age out immediately under the reset boundary — they won't pollute new
  windows. (The in-flight loop should be stopped and restarted under the new
  mechanism; the current iter-4/5 stale data is superseded.)
- `fresh_deficit` is replaced; keep a thin shim or delete with its tests updated.

## 10. Out of scope (tracked elsewhere)

- Recorder→replay divergence root cause / full-observation recording redesign
  (`2026-06-14-recorder-replay-divergence-and-recording-roadmap.md`).
- Data-location move (generate inside repo `runs/`, gitignore, HDD after
  processing) — user-requested, separate small change.
- Arena sims reduction (sims=100) to shorten runtime — deferred optimization.
