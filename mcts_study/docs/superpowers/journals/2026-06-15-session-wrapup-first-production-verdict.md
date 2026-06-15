# Session Wrap-Up — First Production AZ Verdict + Hardened Pipeline (2026-06-14/15)

**Outcome:** the redesigned candidate-self-play AlphaZero loop ran its **first full
production iteration end-to-end on a fully-hardened stack and produced a correct,
trustworthy verdict (iter_6 HOLD)**, then paused cleanly at user request. 122 tests green.

---

## What shipped this session

### 1. Pilot of the candidate-self-play redesign (clean root, tiny scale)
2 cycles validated every redesign mechanism on live data: latest-net self-play
(iter_2 generator = iter_1's trained candidate, not the champion), `gen_iter`
tagging, `new_games>0` (the stale-data-bug column), accumulating window, hold +
invalid gates, graceful stop. Window-reset-on-promotion was the only path not
exercised end-to-end (no promotion; unit-tested). Pilot also caught a real
dashboard liveness bug (read `status.json`, never written mid-self-play → "NOT
RUNNING" while alive; fixed to fall back to `daily_state.json` mtime).

### 2. Adversarial deep inspection → 4 launch blockers fixed
A multi-agent inspection (5 surfaces, each finding skeptic-verified) found **26
confirmed / 26 refuted** faultlines. The 4 must-fixes before the 6-day run:
- **M1** constant seed-base → every iteration replayed the SAME ~1000 boards
  (scientifically worthless). The headline catch — the pilot couldn't surface it.
  Fixed with a per-iteration seed-base term.
- **M2** PUBLISH double-apply → ladder Elo/games corrupted on mid-crash resume.
  Fixed: `record_arena` idempotent keyed by candidate name (`applied_arenas`).
- **M3** arena poison-pills (torn JSONL line / one game's exception wedging the
  gate forever). Fixed: parse-guard + `return_exceptions=True`.
- **M4** silent partial-worker death → trained on a fraction of data. Fixed:
  fail-loud below 0.8×quota + per-worker wait-timeout + returncode checks.

### 3. Production run on `az_loop` (resumed from iter_6)
Surfaced + fixed two more issues only a real run could reveal:
- **Cap-vs-quota mismatch**: hardcoded 6h per-worker cap raced the 1000-game quota
  at ~142 games/hr. Made it a config field, raised to 9h.
- **Deficit-resume duplicate-boards**: resume re-launched workers from the SAME
  seed-bases → would replay the 731 done boards. Fixed with a seed-offset (caught
  before any duplicate flushed). Also added a per-game self-play wall-clock cap
  (600s, measured: normal games ~190s, stragglers 20-60+min) so stragglers
  auto-skip instead of stalling the whole stage.

### 4. Live arena dashboard — rebuilt + made newcomer-legible
The arena (the suspenseful 300-game gate) had no live view. Built one across
3 build+Opus-critique rounds (running winrate, Wilson CI gating the verdict,
clinch/pace tracker, VP-margin persistence, draw-cliff gauge, seat/rotation
fairness, live data-gen card). Then two fresh-no-context Opus agents (25 Qs, then
15×3 Qs) drove an understandability pass: a glossary, labeled chart axes,
disambiguated timeout wording, a fixed seat-bias→rotation metric (the per-board-
seat view falsely showed "0% from seats 1&3"), heartbeat-is-not-realtime
affordances, and an Elo/iterations-table reconciliation (relabeled `champ Elo
after`, added the live in-flight iteration row).

---

## First production verdict

**iter_6: HOLD.** Candidate (iter_5's net) won 48.28% of 300 arena games vs
champion az_iter_1 (cand 98 / champ 105 / draws 97). Below the 65% bar, above the
40%-draw cliff floor → trustworthy HOLD. az_iter_1 stays champion @ Elo 1004.9;
az_iter_6 joined the ladder as a held candidate @ 1004.3. The M2 idempotency guard
confirmed firing (`applied_arenas:['az_iter_6']`). Expected outcome for one AZ
iteration from a converged start — the candidate plays near-identically to the
champion it learned from; promotion is rare per-iteration.

**Measured production stage costs (full scale):** self-play ~142 games/hr →
~7h/1000; train ~35min; arena ~96 games/hr → ~3h/300. Full iteration ≈ ~10-11h.

---

## State at wrap-up

- Loop **PAUSED** before iter_7 (STOP marker present, daily proc exited, GPU idle,
  no iter_7 dir). The current best (champion) is **az_iter_1**.
- Dashboard live at :8099 (independent of the loop), now newcomer-legible.
- **To resume iter_7:** `rm` the STOP marker at
  `/home/chitii/catan_data/runs/v3/az_loop/STOP` and relaunch
  `python3 -m catan_az.daily --loop-root <root> --max-iters 10`. iter_7 gets the
  per-game-timeout + vp_margin + ts fixes live (the arena dashboard's margin/rate
  panels light up from iter_7 onward).

## Known follow-ups (not blocking)
- Self-play ETA edge case at 99%+ (rate includes the dead-start) — cosmetic.
- Per-board-seat margins are unrecoverable for iter_6 (predate the producer fix);
  light up at iter_7.
- The 845 old-mechanism (broken-seed-base) games stay in the accumulating window
  until the first promotion resets the boundary — expected; window system kept
  as-is per user.
