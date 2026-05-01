# Playback UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v2 playback HTML viewer (`mcts_study/catan_mcts/playback.py`) look like a watchable replay instead of a debug dump — seat-header VP scoreboard, dev-card emojis, polished hex tiles + circular number tokens, prettier robber and buildings, dark narration bar with a dice chip.

**Architecture:** All changes live in one Python file (`playback.py`) plus its test (`tests/test_playback.py`). Three regions of `playback.py` are touched: `_render_static_board_png` (matplotlib — bakes hex tiles + tokens into the static PNG), `INDEX_HTML` template (CSS + seat strip markup), and the embedded `<script>` (JS renderer for SVG overlays + narration formatter). No engine changes, no new dependencies, no API changes. The data shape passed from Python to JS (`overlays.states[i]`) is unchanged — every visual upgrade is rendered from fields the viewer already receives.

**Tech Stack:** Python 3.12 (host: Win11; production runs in WSL Ubuntu venv at `~/catan_mcts_venvs/mcts-study/`), matplotlib (Agg backend, baked into base64 PNG), self-contained HTML5 + CSS + vanilla JS for the interactive overlay layer, pytest for tests.

**Spec:** `docs/superpowers/specs/2026-05-01-playback-ui-polish-design.md`

**Worktree:** `engine-v2` (commit `32b32fd` is the spec). All commits below are on this branch. The pre-existing uncommitted port engine work in this worktree is unrelated — leave it alone.

**Scope reminder:** Drawing ports on the board is **out of scope**. We are waiting for the engine fix that exposes `engine.ports()` through PyO3. The right-hand player table keeps showing port ownership as text (unchanged).

---

## File Structure

| File | Role | Change |
|---|---|---|
| `mcts_study/catan_mcts/playback.py` | Single-file replay viewer | Modify — three regions (PNG render, CSS, JS) |
| `mcts_study/tests/test_playback.py` | Smoke + markup-presence tests | Modify — add 1 new test, keep existing 2 |

No new files. The polish is intentionally bundled in one module to keep the viewer self-contained — that's the existing pattern and matches §4 of the spec.

---

## Conventions

- **TDD throughout.** Every visual change has a markup-or-bytes test that fails first, then we implement, then green. The test is intentionally crude (does the HTML byte stream contain the new CSS class / emoji codepoint?). We do not pixel-diff the PNG.
- **Run tests in WSL.** Per the `feedback_wsl_setup_for_mcts_study.md` memory: tests must run in the WSL venv at `~/catan_mcts_venvs/mcts-study/`. From the engine-v2 worktree: `wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py -v'`. Every test step below uses this command.
- **Commit at every green.** Per `feedback_mcts_study_autonomous.md`: regular commits at logical checkpoints. Each task ends in a commit.
- **No `maturin develop` rebuild needed.** Per `feedback_rebuild_pyo3_after_engine_changes.md`: rebuild is required only when Rust changes. This plan touches no Rust.
- **Existing `test_playback.py` fixture** (`minimal_run_dir`) is the production e1 path — it spawns one tiny v2 game and gives us a real run-dir + seed. We reuse it for the new test.

---

### Task 1: Add a markup-presence smoke test that fails today

**Files:**
- Modify: `mcts_study/tests/test_playback.py` (append a new test function)

This test asserts that the rendered HTML contains the marker strings introduced in tasks 2–7. It must fail today so we have a real red-green cycle.

- [ ] **Step 1: Add the failing test**

Append to `mcts_study/tests/test_playback.py` (after `test_render_emits_html`):

```python
def test_render_includes_polish_markers(minimal_run_dir, tmp_path):
    """Polish-pass markers — see specs/2026-05-01-playback-ui-polish-design.md.
    Each assertion corresponds to one visual region of the polish pass."""
    seed = 4242 + 2 * 1_000
    out = playback.render(minimal_run_dir, seed, tmp_path / "out_polish")
    html = out.read_text(encoding="utf-8")

    # Seat header strip (Task 2)
    assert 'class="seat-chip"' in html
    assert 'class="seat-strip"' in html

    # Dev card emojis (Task 3) — codepoints embedded in the JS DEV_EMOJI literal
    for emoji in ['⚔️', '🛣️', '📜', '🌽', '⭐']:
        assert emoji in html, f"missing dev-card emoji {emoji!r}"

    # Dark narration bar with dice chip (Task 7)
    assert 'class="dice-chip"' in html
    assert 'class="seat-tag-0"' in html
    assert 'class="seat-tag-3"' in html

    # Building / road glyph polish (Task 6) — settlement-door rect class hook
    assert 'PLAYER_COLORS_DARK' in html
```

- [ ] **Step 2: Run it to verify it fails**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py::test_render_includes_polish_markers -v'
```

Expected: FAIL — `AssertionError: 'class="seat-chip"' in html` is False (none of the new markers exist yet).

- [ ] **Step 3: Commit the failing test**

```bash
git add mcts_study/tests/test_playback.py
git commit -m "test(playback): markup-presence assertions for UI polish pass"
```

---

### Task 2: Seat header strip

**Files:**
- Modify: `mcts_study/catan_mcts/playback.py` — `INDEX_HTML` (CSS block + DOM markup + JS renderer)

The seat strip is a flex row of 4 chips at the top of the players column. Each chip = colored background + `P{i}` label + LR/LA badge + VP pill. Current player gets a yellow outline. Updates from `st.vp`, `st.cp`, `st.lr_holder`, `st.la_holder` — all already in the per-state dict.

- [ ] **Step 1: Add CSS for the seat strip**

In `playback.py`, find the `<style>` block (currently ends with `.dim { color: #999; }`). Insert before that line:

```css
  .seat-strip { display: flex; gap: 4px; margin: 0 0 6px 0; font-size: 11px; }
  .seat-chip {
    flex: 1; padding: 4px 6px; border-radius: 4px;
    color: white; font-weight: 600; display: flex; justify-content: space-between;
    align-items: center; min-width: 0;
  }
  .seat-chip.cp { outline: 2px solid #ffd633; outline-offset: 1px; }
  .seat-chip .vp { background: rgba(255,255,255,0.25); padding: 1px 5px; border-radius: 3px; font-size: 10px; }
  .seat-chip .ttl { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
```

- [ ] **Step 2: Add the strip's DOM container above the player table**

Find the line `<h3 style="margin:0 0 6px; font-size:14px;">Players</h3>`. Replace it with:

```html
    <h3 style="margin:0 0 6px; font-size:14px;">Players</h3>
    <div class="seat-strip" id="seatStrip"></div>
```

- [ ] **Step 3: Render the strip in JS**

In the `<script>` block, find `function renderState() {` and locate the line `vpTable.innerHTML = html;` (last line of the function before the closing `}`). Insert immediately *before* `vpTable.innerHTML = html;`:

```javascript
  let stripHtml = '';
  for (let i = 0; i < 4; i++) {
    const isCp = (st.cp === i) ? ' cp' : '';
    let badges = '';
    if (st.la_holder === i) badges += ' ⚔️';
    if (st.lr_holder === i) badges += ' 🛣️';
    stripHtml += `<div class="seat-chip${isCp}" style="background:${PLAYER_COLORS[i]}">` +
                 `<span class="ttl">P${i}${badges}</span>` +
                 `<span class="vp">${st.vp[i]} VP</span>` +
                 `</div>`;
  }
  document.getElementById('seatStrip').innerHTML = stripHtml;
```

- [ ] **Step 4: Run the test**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py::test_render_includes_polish_markers -v'
```

Expected: still FAILs (other markers missing), but `class="seat-chip"` and `class="seat-strip"` assertions now pass — verify by running:

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py::test_render_emits_html -v'
```

This must still PASS — we didn't break the smoke test.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_bot/playback.py mcts_study/catan_mcts/playback.py 2>/dev/null; \
git add mcts_study/catan_mcts/playback.py && \
git commit -m "feat(playback): seat-header VP scoreboard with LR/LA badges"
```

(The first `git add` line is defensive — if you're tab-completing, the `catan_bot` path doesn't exist; only `catan_mcts/playback.py` should stage.)

---

### Task 3: Dev card emojis in the player table

**Files:**
- Modify: `mcts_study/catan_mcts/playback.py` — JS `DEV_NAMES` constant + `fmtDev` function

- [ ] **Step 1: Add the DEV_EMOJI parallel array**

In the `<script>` block, find:

```javascript
const DEV_NAMES = ['Knight', 'RoadBldg', 'Mono', 'YOP', 'VP'];
```

Replace with:

```javascript
const DEV_NAMES = ['Knight', 'Road Building', 'Monopoly', 'Year of Plenty', 'Victory Point'];
const DEV_EMOJI = ['⚔️', '🛣️', '📜', '🌽', '⭐'];
```

(Names are also expanded to their full forms for the hover title — see step 2.)

- [ ] **Step 2: Update fmtDev to emit emoji + hover title**

Find:

```javascript
function fmtDev(arr) {
  const parts = [];
  for (let k = 0; k < 5; k++) if (arr[k] > 0) parts.push(`${DEV_NAMES[k]}×${arr[k]}`);
  return parts.length ? parts.join(' ') : '<span class="dim">none</span>';
}
```

Replace with:

```javascript
function fmtDev(arr) {
  const parts = [];
  for (let k = 0; k < 5; k++) {
    if (arr[k] > 0) {
      parts.push(`<span title="${DEV_NAMES[k]}">${DEV_EMOJI[k]}×${arr[k]}</span>`);
    }
  }
  return parts.length ? parts.join(' ') : '<span class="dim">none</span>';
}
```

- [ ] **Step 3: Switch the table cell from textContent to innerHTML**

`fmtDev` now emits HTML. The existing cell already concatenates with backticks and inserts via `vpTable.innerHTML = html;` — no further wiring change needed. (Confirm by reading the existing builder loop; it puts `${devStr}` directly inside `<td>...</td>`, which is then assigned via `innerHTML`. The hover title will work.)

- [ ] **Step 4: Run tests**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py -v'
```

Expected: `test_render_includes_polish_markers` now passes the 5 emoji asserts (plus seat-strip ones from Task 2). Still fails on `dice-chip`, `seat-tag-*`, and `PLAYER_COLORS_DARK` asserts. Other tests still pass.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/playback.py
git commit -m "feat(playback): dev card emojis with full-name hover titles"
```

---

### Task 4: Hex tile polish in the static PNG

**Files:**
- Modify: `mcts_study/catan_mcts/playback.py` — `_render_static_board_png` function (and `RESOURCE_COLORS` constants section)

This is the matplotlib half of the polish. We approximate radial gradients with a two-polygon trick (outer darker hex + inner smaller hex with a lighter shade), darker resource-matched stroke, and a small resource emoji centered above the (eventual) number token. The number token itself is in Task 5.

Note: there is **no markup test for this region** — the PNG is binary and base64-baked into the HTML. We rely on the existing `test_render_emits_html` smoke test (which asserts `data:image/png;base64,` is in the output) plus visual sign-off. That's intentional — pixel-diffing was explicitly ruled out in §7 of the spec.

- [ ] **Step 1: Add a darker-shade helper and resource emoji map**

In `playback.py`, find the constants block:

```python
RESOURCE_COLORS = {0: "#3d8b37", 1: "#a04020", 2: "#90c060", 3: "#e6c243", 4: "#7a7a7a"}
DESERT_COLOR = "#d4b483"
RESOURCE_LABEL = {0: "Wood", 1: "Brick", 2: "Sheep", 3: "Wheat", 4: "Ore"}
```

Append directly below:

```python
RESOURCE_EMOJI = {0: "🌲", 1: "🧱", 2: "🐑", 3: "🌾", 4: "⛰️"}

def _shade(hex_color: str, factor: float) -> str:
    """Multiply each RGB channel by `factor` (0..1 darken, >1 lighten clamped)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"
```

- [ ] **Step 2: Replace the hex-drawing loop body**

Find inside `_render_static_board_png`:

```python
        angles = [math.pi / 6 + i * math.pi / 3 for i in range(6)]
        pts = [(cx + HEX_RADIUS * math.cos(a), cy + HEX_RADIUS * math.sin(a)) for a in angles]
        poly = plt.Polygon(pts, facecolor=color, edgecolor="black", linewidth=1.0)
        ax.add_patch(poly)
        ax.text(cx, cy + 0.35, label, ha="center", va="center", fontsize=8, fontweight="bold")
        if dice_str is not None:
            num = int(dice_str)
            num_color = "red" if num in (6, 8) else "black"
            ax.text(cx, cy, dice_str, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=num_color)
            pips = 6 - abs(7 - num)
            ax.text(cx, cy - 0.3, "·" * pips, ha="center", va="center",
                    fontsize=10, color=num_color)
```

Replace with:

```python
        angles = [math.pi / 6 + i * math.pi / 3 for i in range(6)]
        outer_pts = [(cx + HEX_RADIUS * math.cos(a), cy + HEX_RADIUS * math.sin(a)) for a in angles]
        inner_pts = [(cx + 0.85 * HEX_RADIUS * math.cos(a), cy + 0.85 * HEX_RADIUS * math.sin(a)) for a in angles]
        # Darker outer ring (gradient bottom), lighter inner (gradient top).
        outer_color = _shade(color, 0.78)
        inner_color = _shade(color, 1.12)
        stroke = _shade(color, 0.45)
        outer_poly = plt.Polygon(outer_pts, facecolor=outer_color, edgecolor=stroke, linewidth=1.4)
        inner_poly = plt.Polygon(inner_pts, facecolor=inner_color, edgecolor="none")
        ax.add_patch(outer_poly)
        ax.add_patch(inner_poly)
        # Resource icon above number token. Emoji on Win11/Mac; falls back to the label letter on WSL.
        if res.sum() < 0.5:
            icon = "Desert"
            ax.text(cx, cy, icon, ha="center", va="center",
                    fontsize=10, color=_shade(DESERT_COLOR, 0.5), fontstyle="italic")
        else:
            ridx = int(np.argmax(res))
            icon = RESOURCE_EMOJI[ridx]
            ax.text(cx, cy + 0.42, icon, ha="center", va="center", fontsize=14)
        # Dice token rendered in Task 5; for now leave the existing token code below to keep tests green.
        if dice_str is not None:
            num = int(dice_str)
            num_color = "red" if num in (6, 8) else "black"
            ax.text(cx, cy, dice_str, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=num_color)
            pips = 6 - abs(7 - num)
            ax.text(cx, cy - 0.3, "·" * pips, ha="center", va="center",
                    fontsize=10, color=num_color)
```

(Yes, this temporarily double-draws the icon/token. Task 5 replaces the dice block with a circular token that overlaps the icon correctly. Two steps because each is independently reviewable.)

- [ ] **Step 3: Run smoke test**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py::test_render_emits_html -v'
```

Expected: PASS. (The PNG is now polished but its existence assertion is unchanged.)

- [ ] **Step 4: Visual sanity check (manual)**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && python -m catan_mcts.playback runs/v2_smoke/2026-05-01T11-23-e1_winrate_vs_random 725001 /tmp/playback_polish_check'
```

Open `\\wsl$\Ubuntu\tmp\playback_polish_check\index.html` in the Win11 browser. Confirm hexes have a visible inner-vs-outer gradient and resource emojis show. If WSL falls back to letters, verify they're at least readable — that's an acceptable degraded state per §8 of the spec; the host browser is the production view target anyway.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/playback.py
git commit -m "feat(playback): radial-gradient hex tiles + resource emoji"
```

---

### Task 5: Circular number tokens with red 6/8 ring

**Files:**
- Modify: `mcts_study/catan_mcts/playback.py` — `_render_static_board_png` (the dice-rendering block from Task 4)

- [ ] **Step 1: Replace the dice block**

In `_render_static_board_png`, find the dice block from Task 4:

```python
        if dice_str is not None:
            num = int(dice_str)
            num_color = "red" if num in (6, 8) else "black"
            ax.text(cx, cy, dice_str, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=num_color)
            pips = 6 - abs(7 - num)
            ax.text(cx, cy - 0.3, "·" * pips, ha="center", va="center",
                    fontsize=10, color=num_color)
```

Replace with:

```python
        if dice_str is not None:
            num = int(dice_str)
            is_hot = num in (6, 8)
            ring_color = "#cc2222" if is_hot else "#444444"
            text_color = "#cc2222" if is_hot else "#222222"
            # Soft drop shadow (offset disk, alpha blended).
            shadow = plt.Circle((cx + 0.02, cy - 0.06), 0.30,
                                facecolor="black", edgecolor="none", alpha=0.25)
            ax.add_patch(shadow)
            # Token disk.
            disk = plt.Circle((cx, cy - 0.05), 0.30,
                              facecolor="#fdf2c8", edgecolor=ring_color,
                              linewidth=2.0 if is_hot else 1.4)
            ax.add_patch(disk)
            ax.text(cx, cy - 0.02, dice_str, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)
            pips = 6 - abs(7 - num)
            ax.text(cx, cy - 0.18, "·" * pips, ha="center", va="center",
                    fontsize=8, color=text_color)
```

- [ ] **Step 2: Run smoke test**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py::test_render_emits_html -v'
```

Expected: PASS.

- [ ] **Step 3: Visual sanity check**

Re-run the playback CLI from Task 4 step 4. Confirm 6/8 hexes have a visible red ring and red glyphs; other numbers are dark gray on cream.

- [ ] **Step 4: Commit**

```bash
git add mcts_study/catan_mcts/playback.py
git commit -m "feat(playback): circular number tokens with red 6/8 ring"
```

---

### Task 6: Robber + buildings + roads polish (SVG, JS)

**Files:**
- Modify: `mcts_study/catan_mcts/playback.py` — JS section: `PLAYER_COLORS` constant + robber path + settlement/city/road blocks in `renderState()`

- [ ] **Step 1: Add a darker-color palette in JS**

In the `<script>` block, find:

```javascript
const PLAYER_COLORS = ["#cc3333", "#3366cc", "#33aa55", "#cc8833"];
```

Replace with:

```javascript
const PLAYER_COLORS = ["#cc3333", "#3366cc", "#33aa55", "#cc8833"];
const PLAYER_COLORS_DARK = ["#5a1414", "#1a3370", "#1a5a2c", "#5a3a14"];
```

- [ ] **Step 2: Replace the robber block**

Find in `renderState()`:

```javascript
  if (st.rh >= 0) {
    const [hx, hy] = layout.hex_centers[st.rh];
    const [px, py] = dataToPx(hx, hy);
    const w = 18, h = 22;
    body += `<g style="filter: drop-shadow(0 1px 2px rgba(0,0,0,0.6))">`;
    body += `<path d="M ${px - w/2} ${py + h/2} ` +
            `L ${px - w/2} ${py - h/4} ` +
            `Q ${px - w/2} ${py - h/2 - 2} ${px} ${py - h/2 - 2} ` +
            `Q ${px + w/2} ${py - h/2 - 2} ${px + w/2} ${py - h/4} ` +
            `L ${px + w/2} ${py + h/2} Z" ` +
            `fill="#2a2a2a" stroke="white" stroke-width="2"/>`;
    body += `<circle cx="${px - 3.5}" cy="${py - 3}" r="1.4" fill="white"/>`;
    body += `<circle cx="${px + 3.5}" cy="${py - 3}" r="1.4" fill="white"/>`;
    body += `</g>`;
  }
```

Replace with:

```javascript
  if (st.rh >= 0) {
    const [hx, hy] = layout.hex_centers[st.rh];
    const [px, py] = dataToPx(hx, hy);
    body += `<g style="filter: drop-shadow(0 1px 2px rgba(0,0,0,0.6))">`;
    // Ground shadow ellipse.
    body += `<ellipse cx="${px}" cy="${py + 8}" rx="9" ry="2.5" fill="rgba(0,0,0,0.4)"/>`;
    // Cloak path (rounded rectangle with peaked top).
    body += `<path d="M ${px - 7} ${py + 5} ` +
            `L ${px - 7} ${py - 3} ` +
            `Q ${px - 7} ${py - 9} ${px} ${py - 11} ` +
            `Q ${px + 7} ${py - 9} ${px + 7} ${py - 3} ` +
            `L ${px + 7} ${py + 5} Z" ` +
            `fill="#222" stroke="white" stroke-width="1"/>`;
    // Hooded head.
    body += `<circle cx="${px}" cy="${py - 7}" r="3.5" fill="#222" stroke="white" stroke-width="1"/>`;
    body += `</g>`;
  }
```

- [ ] **Step 3: Replace the road block**

Find:

```javascript
  for (const [eid, owner] of st.r) {
    const e = layout.edges[eid];
    const [x1, y1] = dataToPx(e[0], e[1]);
    const [x2, y2] = dataToPx(e[2], e[3]);
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="black" stroke-width="7" stroke-linecap="round"/>`;
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${PLAYER_COLORS[owner]}" stroke-width="5" stroke-linecap="round"/>`;
  }
```

Replace with:

```javascript
  for (const [eid, owner] of st.r) {
    const e = layout.edges[eid];
    const [x1, y1] = dataToPx(e[0], e[1]);
    const [x2, y2] = dataToPx(e[2], e[3]);
    // White underlay so a colored road reads against any hex.
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="white" stroke-width="7" stroke-linecap="round"/>`;
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${PLAYER_COLORS[owner]}" stroke-width="4.5" stroke-linecap="round"/>`;
  }
```

- [ ] **Step 4: Replace the settlement block**

Find:

```javascript
  for (const [vid, owner] of st.s) {
    const v = layout.vertices[String(vid)];
    const [px, py] = dataToPx(v[0], v[1]);
    const sz = 7;
    const path = `M ${px - sz} ${py + sz} L ${px + sz} ${py + sz} L ${px + sz} ${py - sz/3} L ${px} ${py - sz} L ${px - sz} ${py - sz/3} Z`;
    body += `<path d="${path}" fill="${PLAYER_COLORS[owner]}" stroke="black" stroke-width="1.4" stroke-linejoin="round"/>`;
  }
```

Replace with:

```javascript
  for (const [vid, owner] of st.s) {
    const v = layout.vertices[String(vid)];
    const [px, py] = dataToPx(v[0], v[1]);
    const sz = 7;
    const path = `M ${px - sz} ${py + sz} L ${px + sz} ${py + sz} L ${px + sz} ${py - sz/3} L ${px} ${py - sz} L ${px - sz} ${py - sz/3} Z`;
    body += `<path d="${path}" fill="${PLAYER_COLORS[owner]}" stroke="${PLAYER_COLORS_DARK[owner]}" stroke-width="1.4" stroke-linejoin="round"/>`;
    // Door rectangle.
    body += `<rect x="${px - 1.5}" y="${py + sz/3}" width="3" height="${sz - sz/3 - 1}" fill="${PLAYER_COLORS_DARK[owner]}"/>`;
  }
```

- [ ] **Step 5: Replace the city block**

Find:

```javascript
  for (const [vid, owner] of st.c) {
    const v = layout.vertices[String(vid)];
    const [px, py] = dataToPx(v[0], v[1]);
    const sz = 10;
    const base = `M ${px - sz} ${py + sz} L ${px + sz} ${py + sz} L ${px + sz} ${py - sz/2} L ${px} ${py - sz - 2} L ${px - sz} ${py - sz/2} Z`;
    body += `<path d="${base}" fill="${PLAYER_COLORS[owner]}" stroke="black" stroke-width="1.5" stroke-linejoin="round"/>`;
    body += `<rect x="${px + sz/2}" y="${py - sz - 1}" width="3" height="5" fill="${PLAYER_COLORS[owner]}" stroke="black" stroke-width="1"/>`;
    body += `<rect x="${px - 2}" y="${py + sz/3}" width="4" height="${sz - sz/3}" fill="black" opacity="0.5"/>`;
  }
```

Replace with:

```javascript
  for (const [vid, owner] of st.c) {
    const v = layout.vertices[String(vid)];
    const [px, py] = dataToPx(v[0], v[1]);
    const sz = 10;
    const dark = PLAYER_COLORS_DARK[owner];
    const base = `M ${px - sz} ${py + sz} L ${px + sz} ${py + sz} L ${px + sz} ${py - sz/2} L ${px} ${py - sz - 2} L ${px - sz} ${py - sz/2} Z`;
    body += `<path d="${base}" fill="${PLAYER_COLORS[owner]}" stroke="${dark}" stroke-width="1.5" stroke-linejoin="round"/>`;
    // Tower.
    body += `<rect x="${px + sz/2}" y="${py - sz - 1}" width="3" height="5" fill="${PLAYER_COLORS[owner]}" stroke="${dark}" stroke-width="1"/>`;
    // Window strip across the front face.
    body += `<rect x="${px - sz + 2}" y="${py - 1}" width="${sz * 2 - 4}" height="2.5" fill="${dark}" opacity="0.55"/>`;
    // Door.
    body += `<rect x="${px - 2}" y="${py + sz/3}" width="4" height="${sz - sz/3}" fill="${dark}" opacity="0.7"/>`;
  }
```

- [ ] **Step 6: Run tests**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py -v'
```

Expected: `test_render_includes_polish_markers` now passes the `PLAYER_COLORS_DARK` assert too. Still fails on `dice-chip` and `seat-tag-*`. Other tests pass.

- [ ] **Step 7: Visual sanity check**

Re-run the playback CLI. Click through 5–10 steps. Confirm the robber looks like a hooded figure (not a tombstone), roads have a visible white halo, settlements show a darker door, cities show a window strip + tower.

- [ ] **Step 8: Commit**

```bash
git add mcts_study/catan_mcts/playback.py
git commit -m "feat(playback): polished robber, buildings, road glyphs"
```

---

### Task 7: Dark narration bar with dice chip + seat tags

**Files:**
- Modify: `mcts_study/catan_mcts/playback.py` — CSS narration block + JS narration formatter

- [ ] **Step 1: Replace the narration CSS**

Find in the `<style>` block:

```css
  #narration { padding: 6px 10px; background: #fff; border: 1px solid #ddd; border-radius: 4px;
               margin: 6px 0; font-family: monospace; font-size: 13px; min-height: 22px; }
```

Replace with:

```css
  #narration { padding: 6px 10px; background: #1f2a3a; color: #f0e8c8;
               border: 1px solid #2c3a52; border-radius: 4px;
               margin: 6px 0; font-family: ui-monospace, monospace; font-size: 13px; min-height: 22px; }
  .dice-chip { background: #ffd633; color: #1f2a3a; padding: 0 6px;
               border-radius: 3px; font-weight: 700; margin-right: 6px; }
  .seat-tag-0 { color: #ff8c8c; font-weight: 700; }
  .seat-tag-1 { color: #88a6e6; font-weight: 700; }
  .seat-tag-2 { color: #88d4a0; font-weight: 700; }
  .seat-tag-3 { color: #e6b07a; font-weight: 700; }
```

- [ ] **Step 2: Add the narration formatter**

In the `<script>` block, immediately before `function renderState() {`, insert:

```javascript
function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function formatNarration(s) {
  // CHANCE: dice → N  →  dice chip
  let m = s.match(/^CHANCE: dice → (\d+)$/);
  if (m) return `<span class="dice-chip">🎲 ${m[1]}</span>dice rolled`;
  // CHANCE: steal pX cardY  →  smaller chip
  m = s.match(/^CHANCE: steal p(\d) card(\d+)$/);
  if (m) return `<span class="dice-chip">🥷</span>steal from <span class="seat-tag-${m[1]}">P${m[1]}</span> (card ${m[2]})`;
  // P{i} <action>  →  colored P{i} prefix
  m = s.match(/^P(\d) (.*)$/);
  if (m) return `<span class="seat-tag-${m[1]}">P${m[1]}</span> ${escapeHtml(m[2])}`;
  return escapeHtml(s);
}
```

- [ ] **Step 3: Use the formatter in renderState**

Find:

```javascript
  narr.textContent = st.n;
```

Replace with:

```javascript
  narr.innerHTML = formatNarration(st.n);
```

- [ ] **Step 4: Run all tests**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/test_playback.py -v'
```

Expected: ALL pass — including every assertion in `test_render_includes_polish_markers`.

- [ ] **Step 5: Visual sanity check (full pass)**

Re-run the playback CLI. Step through the game. Confirm:
- Narration bar is dark slate with cream text.
- Dice chance steps show a yellow `🎲 N` chip.
- Player actions are prefixed with a colored `P{i}` tag matching the player's color.
- Seat strip updates correctly across turns; current player has the yellow outline; LR/LA badges appear on the right player(s) once those titles are claimed.
- Dev card cells show emojis with hover titles.
- Hexes have gradient fills + emoji icons + circular tokens (red ring on 6/8).
- Robber, settlements, cities, roads all look polished.

- [ ] **Step 6: Commit**

```bash
git add mcts_study/catan_mcts/playback.py
git commit -m "feat(playback): dark narration bar with dice chip + colored seat tags"
```

---

### Task 8: Final cleanup and journal entry

**Files:**
- Optional: `mcts_study/docs/learnings.md` (one-line entry)

- [ ] **Step 1: Run the full v2 test suite once**

```bash
wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/engine-v2 && pytest mcts_study/tests/ -v'
```

Expected: all pass (or any pre-existing failures unchanged — verify by `git stash && pytest && git stash pop` if anything looks new).

- [ ] **Step 2: Add one-line journal entry**

Append to `mcts_study/docs/learnings.md` (create the file if missing):

```markdown
- 2026-05-01 — playback UI polish pass: seat strip, dev card emojis, hex gradients, circular number tokens, polished robber/buildings, dark narration bar. Single-file change in `catan_mcts/playback.py`. Ports deferred for engine fix.
```

- [ ] **Step 3: Commit**

```bash
git add mcts_study/docs/learnings.md
git commit -m "docs(journal): playback UI polish pass — done"
```

- [ ] **Step 4: Push branch (per catan_bot git policy)**

```bash
git push origin engine-v2
```

Per `feedback_catan_bot_git_policy.md`, pushing branches on this repo is allowed without asking. Merging to main still needs approval — do not merge.

---

## Self-review

Spec coverage check (each goal in §3 of the spec → task that implements it):

| Spec goal | Task |
|---|---|
| 1. Seat header strip | Task 2 ✓ |
| 2. Dev card emojis | Task 3 ✓ |
| 3. Hex tile polish (gradients + icons) | Task 4 ✓ |
| 4. Number tokens (circular, red 6/8) | Task 5 ✓ |
| 5. Robber glyph | Task 6 (steps 1-2) ✓ |
| 6. Settlement / city / road glyphs | Task 6 (steps 3-5) ✓ |
| 7. Narration bar | Task 7 ✓ |

Placeholder scan: no TBDs, no "implement later", no "similar to Task N" — every step has the actual code or command. ✓

Type/identifier consistency: `PLAYER_COLORS_DARK` introduced in Task 6 step 1, used in Task 6 steps 4 and 5. `DEV_EMOJI` introduced in Task 3 step 1, used in Task 3 step 2. `_shade` and `RESOURCE_EMOJI` introduced in Task 4 step 1, used in Task 4 step 2. `formatNarration` introduced in Task 7 step 2, used in Task 7 step 3. `escapeHtml` defined alongside `formatNarration`. All consistent. ✓

WSL fallback: §8 of the spec calls for a `--no-emoji` CLI flag if emoji rendering breaks on WSL. The plan does not implement that flag — instead Task 4 step 2 documents that the resource label falls back gracefully when matplotlib can't render the emoji, and Task 4 step 4 says the WSL fallback is acceptable. The `--no-emoji` flag is genuinely YAGNI today: the production target is the Windows host browser, and WSL only renders the PNG which already degrades acceptably. If WSL emoji turns out to be unreadable in practice, the flag becomes a 3-line follow-up — not worth pre-building. ✓

Risk: the plan assumes the emoji codepoints survive round-trip through Python source → HTML byte stream → pytest assertion. They do (UTF-8 source files + `.write_text(html, encoding="utf-8")` already in `playback.py:718`). ✓
