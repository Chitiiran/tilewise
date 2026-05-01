# Playback UI Polish — Design

**Date:** 2026-05-01
**Worktree:** `engine-v2`
**Target file:** `mcts_study/catan_mcts/playback.py` (single self-contained HTML viewer)
**Status:** design — implementation gated on user approval

## 1. Background

`catan_mcts.playback` emits a single self-contained `index.html` per replayed
seed: a static board PNG + per-step SVG overlays + a side panel showing every
player's hand, dev cards, longest road / largest army, building inventory,
and bank state. The viewer reads everything directly from the v2 engine — no
card-tracker reconstruction.

It works, but it looks like a debug tool: dev cards are plain text
(`Knight×2 RoadBldg×1`), hexes are flat solid colors, number tokens are
loose `8` text + dot pips, the robber is a crude tombstone glyph, and there's
no at-a-glance scoreboard. We want to push it from "debug viewer" to "watchable
replay" without changing what data it reads or how it loads.

## 2. Out of scope

- **Drawing ports on the board.** Ports today are a hardcoded
  `standard_ports()` table in `board.rs:163` with a TODO to randomize in
  Phase 2.7, and there is no Python accessor. We've decided to wait for the
  pending engine fix that exposes ports through PyO3 (likely
  `engine.ports() -> [(v1, v2, kind), ...]`), then add port glyphs in a
  follow-up. The right-hand player table keeps showing per-player port
  ownership as text (unchanged).
- Action-history scrubbing UI changes (slider/buttons/keyboard) — already good.
- Layout math (`_build_layout`, `XLIM/YLIM`, hex/vertex/edge coords) —
  unchanged. All polish is purely visual on top of the same coordinates.
- Accessibility / colorblind palette pass — not requested; can revisit later.
- Performance / file-size optimization — current single-file output is already
  acceptable.

## 3. Goals

1. **Seat header strip** — a 4-cell color-coded scoreboard at the top of the
   players column, current player outlined in yellow, with badges for longest
   road and largest army holders. Lets a viewer track VP momentum without
   reading the table.
2. **Dev card emojis** — replace the text labels in the player table with
   single-glyph emojis: ⚔️ Knight, 🛣️ Road Building, 📜 Monopoly, 🌽 Year of
   Plenty, ⭐ Victory Point.
3. **Hex tile polish** — radial-gradient fills (lighter toward the top to
   suggest light from above), darker resource-matched stroke, and a small
   resource emoji (🌾 🌲 🧱 🐑 ⛰️) inside each hex above the number token.
4. **Number tokens** — circular tan disk with the dice number centered, dot
   pips below. 6 and 8 get a red ring + red glyphs (matches Catan convention).
   Tokens have a soft drop shadow so they pop off the hex.
5. **Robber glyph** — replace the tombstone-ish path with a hooded figure
   silhouette (head circle + cloak path) on a soft elliptical ground shadow.
6. **Settlement / city / road glyphs** — settlements get a darker
   stroke + door rectangle; cities keep their tower but get a window strip;
   roads get a white outline underneath the colored line so they stand out
   against any hex.
7. **Narration bar** — dark slate background with a yellow dice chip
   (`🎲 8`) for chance steps and a colored seat tag (`P0` in player color)
   prefixing player actions. Replaces the current monospace white bar.

## 4. Architecture

The viewer is one Python file that emits one HTML file. Everything is in
`playback.py`. Three regions of that file are touched:

```
playback.py
├── _render_static_board_png()  ← rewrites to draw polished hexes + tokens
│                                  (matplotlib still — it's static, baked into PNG)
├── INDEX_HTML (template literal) ← CSS, seat strip markup, narration markup
└── <script> inside INDEX_HTML    ← renderState(): emoji table cells,
                                     polished SVG building/road/robber paths,
                                     narration formatter, seat-strip update
```

No new modules. No new dependencies. No engine changes.

The data shape passed from Python to JS (`overlays.states[i]`) does not
change — every field needed (vp, lr_holder, la_holder, dev_held, hands,
bank, current player, phase, narration string) already exists. The
narration string for chance dice rolls already starts with `CHANCE: dice → N`,
so the JS can split on that prefix and render the dice chip without any
Python-side changes.

## 5. Components

### 5.1 Seat header strip

A flex row of 4 chips above the player table. Each chip:

- Background = `PLAYER_COLORS[i]` (the existing 4-color palette).
- Text: `P{i}`, optionally `⚔️` if `la_holder == i`, optionally `🛣️` if
  `lr_holder == i`. White text.
- Right-aligned VP pill: white-translucent background, current VP value.
- Current player gets a 2px yellow `outline` (not a border — keeps width
  identical so the strip doesn't reflow).

Updates in `renderState()` from the existing `st.vp`, `st.cp`,
`st.lr_holder`, `st.la_holder`. No new state.

### 5.2 Dev card emojis

Replace `DEV_NAMES = ['Knight', 'RoadBldg', 'Mono', 'YOP', 'VP']` with a
parallel `DEV_EMOJI = ['⚔️', '🛣️', '📜', '🌽', '⭐']`. `fmtDev()` becomes:

```js
for (let k = 0; k < 5; k++) if (arr[k] > 0) parts.push(`${DEV_EMOJI[k]}×${arr[k]}`);
```

Hover title (`<span title="Knight ×2">⚔️×2</span>`) keeps the text label
discoverable for anyone who doesn't recognize an emoji.

### 5.3 Hex tile polish (matplotlib)

`_render_static_board_png` currently fills hexes with a single
`facecolor=RESOURCE_COLORS[ridx]`. Polish:

- Add a `RadialGradient`-equivalent: matplotlib doesn't have native gradients,
  so we approximate by drawing each hex as an outer polygon + a slightly
  smaller inner polygon with a lighter shade. Two-stop gradient suffices.
- Stroke color: a darker shade of the resource color (e.g. multiply each
  RGB channel by 0.45) instead of plain black, so each tile has its own
  edge identity.
- Resource emoji placed at `cy + 0.42` (above where the number token sits).
  Matplotlib renders emoji through the host's color emoji font, which works
  on the Win11 host but is unreliable in our WSL Ubuntu venv — see §8 for
  the fallback.
- Number tokens become a filled circle (`Circle(radius=0.32)`) with cream
  fill (`#fdf2c8`) and a red edge for 6/8, dark gray edge otherwise. Number
  text + pip row centered inside. Drop shadow approximated with a second
  circle offset by `(0, -0.04)` and `alpha=0.25` painted first.

### 5.4 Robber glyph (SVG, JS)

Replace the current tombstone path with a two-element silhouette:

```js
// shadow ellipse on the ground, then cloak path, then head circle
body += `<ellipse cx="${px}" cy="${py + 8}" rx="9" ry="2.5" fill="rgba(0,0,0,0.4)"/>`;
body += `<path d="M ${px-7} ${py+5} L ${px-7} ${py-3} Q ${px-7} ${py-9} ${px} ${py-11}
                  Q ${px+7} ${py-9} ${px+7} ${py-3} L ${px+7} ${py+5} Z"
              fill="#222" stroke="white" stroke-width="1"/>`;
body += `<circle cx="${px}" cy="${py-7}" r="3.5" fill="#222" stroke="white" stroke-width="1"/>`;
```

Same `drop-shadow` filter as today. Replaces the existing
"tombstone + two eye dots" block in `renderState()`.

### 5.5 Building / road glyphs (SVG, JS)

- **Roads** — keep the existing two-line trick (black underlay + colored
  top), but increase the underlay to white with `stroke-width="6"` so a
  red road on a brick hex still reads. Colored top stays at width 4. Both
  lines keep `stroke-linecap="round"`.
- **Settlements** — same house path, but stroke becomes a dark shade of the
  player's color (precomputed `PLAYER_COLORS_DARK[i]`) instead of black, plus
  a small `<rect>` "door" centered at the bottom.
- **Cities** — keep the existing house-with-tower path, add a dark-rectangle
  "window strip" across the front face for visual weight. Stroke darker.

### 5.6 Narration bar

CSS:

```css
#narration {
  background: #1f2a3a;
  color: #f0e8c8;
  border: 1px solid #2c3a52;
  font-family: ui-monospace, monospace;
}
.dice-chip {
  background: #ffd633;
  color: #1f2a3a;
  padding: 0 6px;
  border-radius: 3px;
  font-weight: 700;
  margin-right: 6px;
}
.seat-tag-0 { color: #ff8c8c; font-weight: 700; }
.seat-tag-1 { color: #88a6e6; font-weight: 700; }
.seat-tag-2 { color: #88d4a0; font-weight: 700; }
.seat-tag-3 { color: #e6b07a; font-weight: 700; }
```

JS formatter (replaces `narr.textContent = st.n`):

```js
function formatNarration(s) {
  // CHANCE: dice → N  →  <chip>🎲 N</chip>
  let m = s.match(/^CHANCE: dice → (\d+)$/);
  if (m) return `<span class="dice-chip">🎲 ${m[1]}</span>dice rolled`;
  // P{i} <action>  →  <colored P{i}> <action>
  m = s.match(/^P(\d) (.*)$/);
  if (m) return `<span class="seat-tag-${m[1]}">P${m[1]}</span> ${escapeHtml(m[2])}`;
  return escapeHtml(s);
}
narr.innerHTML = formatNarration(st.n);
```

`escapeHtml` is a 4-line helper; the existing narration strings come from
`_action_desc` and contain only `<`, `>`, `→`, `↔` — safe to escape.

## 6. Data flow

Unchanged. Every visual change is rendered from data we already pass:

| Visual | Source field |
|---|---|
| Seat strip VP, current-player, LR/LA badges | `st.vp`, `st.cp`, `st.lr_holder`, `st.la_holder` |
| Dev card emojis | `st.dev_held[i]` (5-vec per player) |
| Hex tiles + number tokens | from engine in `_render_static_board_png`, baked into PNG |
| Robber position | `st.rh` |
| Buildings / roads | `st.s`, `st.c`, `st.r` |
| Narration | `st.n` |

## 7. Testing

There is one playback test today: `mcts_study/tests/test_playback.py`. It runs
the full `render()` end-to-end on a small parquet fixture and asserts the
HTML file is created. Polish-pass additions to that test:

1. **Smoke** (existing) — still produces a non-empty `index.html`.
2. **Markup presence** — assert the rendered HTML contains:
   - `class="seat-chip"` (seat strip rendered)
   - the 5 dev-card emoji codepoints (`⚔️`, `🛣️`, `📜`, `🌽`, `⭐`)
   - `class="dice-chip"` (narration formatter wired in)
   - `class="seat-tag-0"` (player-tag CSS class is in the stylesheet)
3. **No regressions** — existing assertions on board PNG existence, step
   count, and that every state field round-trips through JSON unchanged.

The board PNG is binary; we don't pixel-diff it. Visual sign-off is by
opening `index.html` for one production seed and eyeballing it — same
manual check we already do today.

## 8. Risks & open questions

- **Emoji rendering in matplotlib.** The hex resource emojis (🌾 🌲 🧱 🐑 ⛰️)
  rely on the system having a color emoji font matplotlib can find. On the
  Win11 host it works through Segoe UI Emoji; in the WSL Ubuntu venv we use
  for production runs, color emoji support is patchy. **Mitigation:**
  fall back to a single-letter glyph (`W`/`B`/`Sh`/`Wh`/`Or`) styled in white
  if the emoji glyph fails to render, and the playback CLI accepts a
  `--no-emoji` flag for headless runs. We bake the PNG once per seed, so the
  fallback only matters when running the polish pass on WSL.
- **Year of Plenty / VP emoji choice.** 🌽 and ⭐ are not iconic for these
  cards. We picked them because no Unicode glyph cleanly represents
  "any-2-resources" or "victory point". Easy to swap (single string change).
- **Phase 2.7 port randomization.** When that lands, we add a `ports` field
  to the per-state dict and draw glyphs in `renderState()`. Today's design
  doesn't paint us into a corner — port glyphs go in the same SVG layer
  used for buildings and the robber.

## 9. Implementation plan handoff

This design becomes a single-file change to `mcts_study/catan_mcts/playback.py`
plus an extension of `mcts_study/tests/test_playback.py`. No engine changes,
no new dependencies, no API changes. The next step is to invoke the
`writing-plans` skill to produce a TDD-ordered task list (tests first per
project convention, then implementation per visual region).
