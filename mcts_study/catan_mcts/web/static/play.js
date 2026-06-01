const PLAY = document.getElementById('tab-play');
let BOTS = null;

async function initLobby() {
  BOTS = await (await fetch('/api/bots')).json();
  renderLobby();
}

// Default opponent: PureGnn loading the "Cell 6" checkpoint — the strongest
// FAST bot for full Catan per our tournament data (~54% in 4-PureGnn, CPU-only
// so it doesn't touch the GPU). LookaheadV3 wins more (~70%) but is slow.
const DEFAULT_BOT_TYPE = 'PureGnn';

// Find the Cell 6 epoch-10 checkpoint among the discovered .pt files. Matched
// by distinctive path fragments so a runs-dir reorg doesn't break it; falls
// back to the first checkpoint, then to null (caller degrades the default).
function defaultCheckpointPath() {
  const cks = BOTS.checkpoints || [];
  if (!cks.length) return null;
  const cell6 = cks.find(c =>
    /06_cand11_cand8_cand10/.test(c.label) && /checkpoint_epoch10\.pt$/.test(c.label));
  return (cell6 || cks[0]).path;
}

function botSelect(seat) {
  // Default the type to PureGnn (Cell 6) when a checkpoint is available;
  // otherwise leave the first type selected so the lobby still works.
  const haveCkpt = (BOTS.checkpoints || []).length > 0;
  const defType = haveCkpt ? DEFAULT_BOT_TYPE : (BOTS.types[0] && BOTS.types[0].id);
  const opts = BOTS.types.map(t =>
    `<option value="${t.id}"${t.id === defType ? ' selected' : ''}>${t.label}</option>`).join('');
  return `<select class="bot-type" data-seat="${seat}">${opts}</select>
          <select class="bot-ckpt" data-seat="${seat}" style="display:none"></select>`;
}

function renderLobby() {
  let rows = '';
  for (let s = 0; s < 4; s++) {
    rows += `<tr>
      <td><label><input type="radio" name="human" value="${s}" ${s===0?'checked':''}> Seat P${s}</label></td>
      <td class="seat-bot" data-seat="${s}">${botSelect(s)}</td></tr>`;
  }
  PLAY.innerHTML = `
    <div class="panel" style="max-width:520px">
      <h2>New game</h2>
      <table>${rows}</table>
      <div style="margin:10px 0">
        VP target <select id="vp"><option value="10" selected>10 (full)</option><option value="5">5 (short)</option></select>
        &nbsp; <label><input type="checkbox" id="bonuses" checked> bonuses (LR/LA +2)</label>
        &nbsp; <label><input type="checkbox" id="showBank" checked> show bank</label>
        &nbsp; seed <input id="seed" type="number" placeholder="random" style="width:90px">
      </div>
      <button class="primary" id="start">Start Game</button>
      <span id="lobby-err" style="color:#c33"></span>
    </div>`;
  syncHumanSeat();
  PLAY.querySelectorAll('input[name=human]').forEach(r => r.onchange = syncHumanSeat);
  PLAY.querySelectorAll('.bot-type').forEach(sel => sel.onchange = () => syncCkpt(sel));
  // Populate each seat's checkpoint dropdown for its (defaulted) bot type, so
  // a PureGnn default shows its Cell 6 checkpoint pre-selected.
  PLAY.querySelectorAll('.bot-type').forEach(sel => syncCkpt(sel));
  document.getElementById('start').onclick = onStart;
}

function syncHumanSeat() {
  const human = +PLAY.querySelector('input[name=human]:checked').value;
  for (let s = 0; s < 4; s++) {
    const cell = PLAY.querySelector(`.seat-bot[data-seat="${s}"]`);
    cell.style.visibility = (s === human) ? 'hidden' : 'visible';
  }
}

// Build the checkpoint dropdown options once, grouped by run directory via
// <optgroup>, labeled with the dir-qualified relative path so the many
// same-named files (dozens of checkpoint_best.pt) are distinguishable.
function checkpointOptionsHtml() {
  const cks = BOTS.checkpoints || [];
  if (!cks.length) return '<option value="">(no .pt found)</option>';
  const groups = new Map();   // dir -> [checkpoint, ...]
  for (const c of cks) {
    const g = c.dir || '(top level)';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(c);
  }
  let html = '';
  for (const [dir, items] of groups) {
    const opts = items.map(c => {
      // Within a group, show just the filename; the group header carries the dir.
      const shown = c.name || c.label;
      return `<option value="${c.path}" title="${escapeHtml(c.label)}">${escapeHtml(shown)}</option>`;
    }).join('');
    html += `<optgroup label="${escapeHtml(dir)}">${opts}</optgroup>`;
  }
  return html;
}

function syncCkpt(sel) {
  const seat = sel.dataset.seat;
  const ck = PLAY.querySelector(`.bot-ckpt[data-seat="${seat}"]`);
  const type = BOTS.types.find(t => t.id === sel.value);
  if (type && type.needs_checkpoint) {
    ck.innerHTML = checkpointOptionsHtml();
    // Pre-select the strong default (Cell 6) when this dropdown is first shown.
    const def = defaultCheckpointPath();
    if (def) ck.value = def;
    ck.style.display = '';
  } else {
    ck.style.display = 'none';
  }
}

async function onStart() {
  const human = +PLAY.querySelector('input[name=human]:checked').value;
  const seats = {};
  for (let s = 0; s < 4; s++) {
    if (s === human) continue;
    const type = PLAY.querySelector(`.bot-type[data-seat="${s}"]`).value;
    const spec = { type };
    const ck = PLAY.querySelector(`.bot-ckpt[data-seat="${s}"]`);
    if (ck.style.display !== 'none' && ck.value) spec.checkpoint = ck.value;
    seats[s] = spec;
  }
  const seedVal = document.getElementById('seed').value;
  // The bank is purely client-display (always present in state); carry the
  // toggle to the game screen via a global instead of the POST body.
  window._showBank = document.getElementById('showBank').checked;
  const body = {
    human_seat: human, seats,
    rules: { vp_target: +document.getElementById('vp').value,
             bonuses: document.getElementById('bonuses').checked },
    seed: seedVal === '' ? null : +seedVal,
  };
  const r = await fetch('/api/games', { method: 'POST', headers: {'Content-Type':'application/json'},
                                        body: JSON.stringify(body) });
  if (!r.ok) { document.getElementById('lobby-err').textContent = (await r.json()).detail; return; }
  startGame(await r.json());   // defined in Task 18
}

initLobby();

// ---- Game screen ----------------------------------------------------------
let G = null;  // { gid, layout, png, state, _sse }

function startGame(body) {
  G = { gid: body.game_id, layout: body.board.layout, png: body.board.png_b64,
        state: body.state };
  // Client-only bank-strip visibility (default true), set by the lobby toggle.
  G.showBank = window._showBank !== false;
  PLAY.innerHTML = `
    <div class="game-screen">
      <div class="panel board-col">
        <div id="boardWrap">
          <img id="board" src="data:image/png;base64,${G.png}">
          <svg id="overlay" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
      </div>
      <div class="side-col">
        <div class="panel side-players">
          <div id="status"></div>
          <div id="players"></div>
          <div id="bank"></div>
        </div>
        <div class="panel side-log">
          <h3 class="side-log-title">Log</h3>
          <div id="log"></div>
        </div>
        <div class="panel side-actions">
          <div id="actionBar"></div>
        </div>
      </div>
    </div>`;
  document.getElementById('board').addEventListener('load', renderGame);
  window.addEventListener('resize', onResize);
  applyState(G.state);
}

// Re-map the overlay whenever the board's display size changes (the overlay
// viewBox is derived from img.clientWidth/clientHeight).
function onResize() { if (G && G.state) renderGame(); }

const PLAYER_COLORS = ["#cc3333", "#3366cc", "#33aa55", "#cc8833"];
// Darker strokes for the house/city glyphs (ported from playback.py).
const PLAYER_COLORS_DARK = ["#5a1414", "#1a3370", "#1a5a2c", "#5a3a14"];
const RES = ['🪵','🧱','🐑','🌾','⛰️'];
// Letter fallback for environments without a color-emoji font (mirrors
// board_layout.RESOURCE_LETTER). Wrapped so a missing glyph still reads.
const RES_LETTER = ['W', 'B', 'S', 'Wh', 'O'];
function res(r) { return `<span title="${RES_LETTER[r]}">${RES[r]}</span>`; }

// Dev card glyphs/names — index matches the dev_held slot order
// (Knight, RoadBuilding, Monopoly, YearOfPlenty, VictoryPoint).
const DEV_EMOJI = ['⚔️','🛣️','📜','🌽','⭐'];
const DEV_NAMES = ['Knight','Road Building','Monopoly','Year of Plenty','Victory Point'];

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
}

function formatNarration(s) {
  // Ported (simplified) from playback.py formatNarration.
  let m = s.match(/^CHANCE: dice → (\d+)$/);
  if (m) return `🎲 ${m[1]} dice rolled`;
  m = s.match(/^CHANCE: steal p(\d) card(\d+)$/);
  if (m) return `🥷 steal from <span class="seat-${m[1]}">P${m[1]}</span> (card ${m[2]})`;
  m = s.match(/^P(\d) (.*)$/);
  if (m) return `<span class="seat-${m[1]}">P${m[1]}</span> ${escapeHtml(m[2])}`;
  return escapeHtml(s);
}

function dataToPx(x, y) {
  const img = document.getElementById('board');
  const w = img.clientWidth, h = img.clientHeight;
  const [x0,x1] = G.layout.xlim, [y0,y1] = G.layout.ylim;
  return [((x-x0)/(x1-x0))*w, h-((y-y0)/(y1-y0))*h];
}

function applyStateNoStream(st) {
  G.state = st;
  renderGame();
  if (st.narration) {
    const log = document.getElementById('log');
    log.innerHTML += `<div>${formatNarration(st.narration)}</div>`;
    log.scrollTop = log.scrollHeight;
  }
  if (st.status === 'trade_offer') {
    showTradeModal(st.trade_offer);
  } else {
    // Any non-offer state cancels a pending auto-reject and clears stale modals.
    clearTradeTimer();
    document.querySelectorAll('.modal-bg').forEach(m => m.remove());
  }
  if (st.status === 'game_over' && !G._celebrated) {
    G._celebrated = true;       // fire once per game, not on every re-render
    celebrateGameOver(st);
  }
  renderActionBar(st);
}

function applyState(st) {
  applyStateNoStream(st);
  maybeStreamBots(st);
}

function renderGame() {
  if (!G || !G.state) return;
  const st = G.state.state;       // the serialized board sub-object
  const svg = document.getElementById('overlay');
  const img = document.getElementById('board');
  svg.setAttribute('viewBox', `0 0 ${img.clientWidth} ${img.clientHeight}`);
  let body = '';
  // Robber.
  if (st.rh >= 0) {
    const [hx,hy] = G.layout.hex_centers[st.rh]; const [px,py] = dataToPx(hx,hy);
    body += `<circle cx="${px}" cy="${py-10}" r="6" fill="#222" stroke="#fff" stroke-width="1.5"/>`;
  }
  // Roads.
  for (const [eid,o] of st.r) {
    const e = G.layout.edges[eid];
    const [x1,y1] = dataToPx(e[0],e[1]); const [x2,y2] = dataToPx(e[2],e[3]);
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#fff" stroke-width="7" stroke-linecap="round"/>`;
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${PLAYER_COLORS[o]}" stroke-width="4.5" stroke-linecap="round"/>`;
  }
  // Last-move glow (cyan halo) under the piece, so it's clear "the game is
  // here right now". Distinct from the amber clickable-target markers.
  body += lastMoveGlowSvg(G.state);
  // Settlements = small houses (pentagon + door). Ported from playback.py.
  for (const [vid,o] of st.s) {
    const v = G.layout.vertices[String(vid)]; const [px,py] = dataToPx(v[0],v[1]);
    body += settlementGlyph(px, py, o);
  }
  // Cities = bigger houses with a tower + window strip. Ported from playback.py.
  for (const [vid,o] of st.c) {
    const v = G.layout.vertices[String(vid)]; const [px,py] = dataToPx(v[0],v[1]);
    body += cityGlyph(px, py, o);
  }
  // Clickable spatial targets when it's your turn (Task 19 fills click handlers).
  body += spatialTargetsSvg(G.state);
  svg.innerHTML = body;
  renderPlayers(G.state);
  renderStatus(G.state);
}

// Settlement glyph: a small house (square base + peaked roof) with a door.
function settlementGlyph(px, py, o) {
  const sz = 11;
  const path = `M ${px-sz} ${py+sz} L ${px+sz} ${py+sz} L ${px+sz} ${py-sz/3} `
             + `L ${px} ${py-sz} L ${px-sz} ${py-sz/3} Z`;
  const dark = PLAYER_COLORS_DARK[o];
  return `<path d="${path}" fill="${PLAYER_COLORS[o]}" stroke="${dark}" stroke-width="1.8" stroke-linejoin="round"/>`
       + `<rect x="${px-2.5}" y="${py+sz/3}" width="5" height="${sz-sz/3-1}" fill="${dark}"/>`;
}

// City glyph: a bigger house with a corner tower, a horizontal window strip,
// and a door — unmistakably larger/more detailed than a settlement.
function cityGlyph(px, py, o) {
  const sz = 14;
  const dark = PLAYER_COLORS_DARK[o];
  const base = `M ${px-sz} ${py+sz} L ${px+sz} ${py+sz} L ${px+sz} ${py-sz/2} `
             + `L ${px} ${py-sz-3} L ${px-sz} ${py-sz/2} Z`;
  return `<path d="${base}" fill="${PLAYER_COLORS[o]}" stroke="${dark}" stroke-width="1.8" stroke-linejoin="round"/>`
       + `<rect x="${px+sz/2}" y="${py-sz-1}" width="4" height="7" fill="${PLAYER_COLORS[o]}" stroke="${dark}" stroke-width="1.2"/>`
       + `<rect x="${px-sz+3}" y="${py-1}" width="${sz*2-6}" height="3.5" fill="${dark}" opacity="0.55"/>`
       + `<rect x="${px-3}" y="${py+sz/3}" width="6" height="${sz-sz/3}" fill="${dark}" opacity="0.7"/>`;
}

// Map a raw action id to {kind,target} for spatial moves, mirroring the
// server's action_decode ranges. Non-spatial actions return null.
function actionSpatialTarget(a) {
  a = Number(a);
  if (a >= 0 && a < 54)    return { kind: 'settlement', target: a };
  if (a >= 54 && a < 108)  return { kind: 'city', target: a - 54 };
  if (a >= 108 && a < 180) return { kind: 'road', target: a - 108 };
  if (a >= 180 && a < 199) return { kind: 'robber', target: a - 180 };
  return null;
}

// Draw a pulsing cyan halo at the last applied move's spatial target.
function lastMoveGlowSvg(g) {
  const la = g.last_action;
  if (!la) return '';
  const t = actionSpatialTarget(la.action);
  if (!t) return '';
  let px, py;
  if (t.kind === 'road') {
    const e = G.layout.edges[t.target]; if (!e) return '';
    [px, py] = dataToPx((e[0]+e[2])/2, (e[1]+e[3])/2);
  } else if (t.kind === 'robber') {
    const c = G.layout.hex_centers[t.target]; if (!c) return '';
    [px, py] = dataToPx(c[0], c[1]);
  } else {
    const v = G.layout.vertices[String(t.target)]; if (!v) return '';
    [px, py] = dataToPx(v[0], v[1]);
  }
  return `<circle class="last-move-glow" cx="${px}" cy="${py}" r="17" fill="#33e0ff" `
       + `fill-opacity="0.28" stroke="#33e0ff" stroke-width="2.5" stroke-opacity="0.8"/>`;
}

function renderPlayers(g) {
  const st = g.state; let rows = '';
  for (let i = 0; i < 4; i++) {
    const h = st.hands[i];
    const hand = h.breakdown.map((n,r) => n>0?`${res(r)}${n}`:'').filter(Boolean).join(' ');
    const isCp = g.current_player === i;
    const cp = isCp ? '▶ ' : '';
    // Longest Road / Largest Army badges. The +2 VP each is ALREADY included in
    // st.vp[i] by the engine; these badges show who holds them (and why), since
    // otherwise the bonus looks invisible. Show the length / knight count too.
    let bonus = '';
    if (st.lr_holder === i) bonus += `<span class="badge badge-lr" title="Longest Road (+2 VP)">🛣️ LR ${st.lr_len[i]}</span>`;
    if (st.la_holder === i) bonus += `<span class="badge badge-la" title="Largest Army (+2 VP)">⚔️ LA ${st.knights[i]}</span>`;
    rows += `<tr class="${isCp ? 'cp-row' : ''}"><td class="seat-${i}"><b>${cp}${g.seat_names[i]}</b></td>
             <td>${st.vp[i]} VP</td><td>${bonus || '—'}</td><td>${hand||'—'}</td></tr>`;
  }
  document.getElementById('players').innerHTML =
    `<table><tr><th>seat</th><th>VP</th><th>bonus</th><th>hand</th></tr>${rows}</table>`;
  renderBank(g);
}

// Compact bank strip: "🏦 Bank: 🪵19 🧱19 🐑19 🌾19 ⛰️19". Client-only display
// gated on G.showBank; hidden (empty) when the lobby toggle is off.
function renderBank(g) {
  const el = document.getElementById('bank');
  if (!el) return;
  if (!G.showBank) { el.innerHTML = ''; return; }
  const bank = g.state.bank || [];
  const cells = bank.map((n, r) => `${res(r)}${n}`).join(' ');
  el.innerHTML = `<div class="bank-strip">🏦 Bank: ${cells}</div>`;
}

function renderStatus(g) {
  const map = { your_turn: 'Your turn', bot_thinking: 'Bot thinking…',
                trade_offer: 'Trade offer', game_over: 'Game over', error: 'Error' };
  let txt = map[g.status] || g.status;
  let cls = '';   // extra class for attention-grabbing phases
  if (g.status === 'game_over' && g.returns) {
    const w = g.returns.indexOf(1);
    txt = (w === g.human_seat) ? 'You win 🎉' : `${g.seat_names[w]} wins`;
  }
  // Make the robber / discard prompts unmistakable — these are the moments the
  // game looks "hung" if the player doesn't realize it's waiting on them.
  if (g.status === 'your_turn' && g.legal_actions) {
    const kinds = new Set(g.legal_actions.map(a => a.kind));
    if (kinds.has('move_robber')) {
      txt = '🦹 Move the robber — click a highlighted hex';
      cls = 'status-alert';
    } else if (kinds.has('discard')) {
      txt = '⚠️ You rolled a 7 — discard cards (pick below)';
      cls = 'status-alert';
    }
  }
  const el = document.getElementById('status');
  el.className = cls;
  el.innerHTML = `<b>${txt}</b>`;
}

// ---- Interaction (Task 19) ------------------------------------------------
async function postAction(actionId) {
  const r = await fetch(`/api/games/${G.gid}/action`,
    { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ action: actionId }) });
  if (r.status === 409) { const s = await fetch(`/api/games/${G.gid}/state`); applyState(await s.json()); return; }
  applyState(await r.json());
}

function spatialTargetsSvg(g) {
  if (g.status !== 'your_turn' || !g.legal_actions) return '';
  let out = '';
  for (const a of g.legal_actions) {
    if (a.target === null) continue;
    if (a.kind === 'move_robber') {
      // Robber targets get a big, pulsing, unmistakable marker covering the
      // whole hex with a 🦹 glyph — so it's obvious the game is waiting on you
      // to move the robber (otherwise it reads as "hung").
      const c = G.layout.hex_centers[a.target];
      const [px,py] = dataToPx(c[0],c[1]);
      out += `<circle class="robber-target clickable" cx="${px}" cy="${py}" r="26"
               fill="#ff5252" fill-opacity="0.30" stroke="#b00020" stroke-width="3"
               style="pointer-events:all" onclick="postAction(${a.id})"/>`;
      out += `<text x="${px}" y="${py+6}" text-anchor="middle" font-size="20"
               style="pointer-events:none">🦹</text>`;
      continue;
    }
    let px, py;
    if (a.kind === 'build_road') {
      const e = G.layout.edges[a.target]; [px,py] = dataToPx((e[0]+e[2])/2,(e[1]+e[3])/2);
    } else {
      const v = G.layout.vertices[String(a.target)]; [px,py] = dataToPx(v[0],v[1]);
    }
    out += `<circle class="clickable" cx="${px}" cy="${py}" r="10" fill="#ffd633" fill-opacity="0.5"
             stroke="#c90" stroke-width="2" style="pointer-events:all" onclick="postAction(${a.id})"/>`;
  }
  return out;
}

// Decode a propose_trade action id (260..279) into [give, get] resource
// indices, mirroring the server's encoding: idx = id-260, give = idx//4, and
// get = the (idx%4)-th resource != give.
function decodeProposeTrade(id) {
  const idx = id - 260, give = Math.floor(idx / 4);
  const others = [0, 1, 2, 3, 4].filter(r => r !== give);
  const get = others[idx % 4];
  return [give, get];
}

// Decode a trade_bank action id (206..225) into [give, get] resource indices —
// same formula as propose_trade but base 206 (mirrors serializers.action_desc).
function decodeBankTrade(id) {
  const idx = id - 206, give = Math.floor(idx / 4);
  const others = [0, 1, 2, 3, 4].filter(r => r !== give);
  const get = others[idx % 4];
  return [give, get];
}

function renderActionBar(g) {
  const bar = document.getElementById('actionBar');
  if (g.status !== 'your_turn' || !g.legal_actions) { bar.innerHTML = ''; return; }
  // Non-spatial actions become buttons; spatial ones are board clicks.
  // propose_trade + trade_bank become give→get grids; play_dev becomes the
  // clickable dev-card hand. Those kinds are NOT plain buttons.
  const NON_SPATIAL = new Set(['roll','end_turn','buy_dev','discard']);
  const seen = new Map();
  const proposeIds = [];      // legal propose_trade action ids
  const bankIds = [];         // legal trade_bank action ids
  const devIds = [];          // legal play_dev action ids
  for (const a of g.legal_actions) {
    if (a.kind === 'propose_trade') { proposeIds.push(a.id); continue; }
    if (a.kind === 'trade_bank')    { bankIds.push(a.id); continue; }
    if (a.kind === 'play_dev')      { devIds.push(a.id); continue; }
    if (!NON_SPATIAL.has(a.kind)) continue;
    if (!seen.has(a.id)) seen.set(a.id, a);
  }
  let html = [...seen.values()]
    .map(a => `<button onclick="postAction(${a.id})">${a.label}</button>`).join(' ');
  const bankGridLabel = bankIds.length
    ? `<div class="trade-grid-label">Bank Trade</div>` : '';
  const proposeGridLabel = proposeIds.length
    ? `<div class="trade-grid-label">Propose Trade</div>` : '';
  bar.innerHTML = `<div class="action-row">${html}</div>` +
                  `<div id="devCards"></div>` +
                  bankGridLabel +
                  `<div id="bankGrid" class="trade-grid-wrap"></div>` +
                  proposeGridLabel +
                  `<div id="tradeGrid" class="trade-grid-wrap"></div>`;
  renderDevCards(g, new Set(devIds));
  // Build the trade grids from the legal ids (empty if none).
  buildTradeGrid('bankGrid', bankIds, decodeBankTrade, postAction);
  buildTradeGrid('tradeGrid', proposeIds, decodeProposeTrade, postTradeFromGrid);
}

// Render the human's held dev cards as clickable little cards. Knight (227) and
// Road Building (228) play directly; Monopoly (229..233) and Year of Plenty
// (234..258) expand inline resource pickers; VP (slot 4) is never clickable.
// A card type is "playable now" iff at least one of its legal play_dev ids is
// in legalDevIds.
function renderDevCards(g, legalDevIds) {
  const el = document.getElementById('devCards');
  if (!el) return;
  const st = g.state;
  const seat = g.human_seat;
  const held = (st.dev_held && st.dev_held[seat]) || [0, 0, 0, 0, 0];
  const total = held.reduce((a, b) => a + b, 0);
  if (!total) { el.innerHTML = ''; return; }
  let cards = '';
  for (let slot = 0; slot < 5; slot++) {
    const n = held[slot];
    if (n <= 0) continue;
    const label = `${DEV_EMOJI[slot]} ${DEV_NAMES[slot]} <span class="dev-count">×${n}</span>`;
    if (slot === 4) {                       // Victory Point — never playable
      cards += `<div class="dev-card dev-vp" title="auto-counted">${label}</div>`;
      continue;
    }
    if (slot === 0 || slot === 1) {         // Knight / Road Building — direct
      const id = slot === 0 ? 227 : 228;
      const ok = legalDevIds.has(id);
      cards += ok
        ? `<div class="dev-card" onclick="postAction(${id})">${label}</div>`
        : `<div class="dev-card dev-dim" title="can't play now">${label}</div>`;
      continue;
    }
    if (slot === 2) {                       // Monopoly — pick 1 resource (229..233)
      const anyLegal = [0,1,2,3,4].some(r => legalDevIds.has(229 + r));
      cards += anyLegal
        ? `<div class="dev-card" onclick="toggleDevPicker('mono')">${label}</div>`
        : `<div class="dev-card dev-dim" title="can't play now">${label}</div>`;
      continue;
    }
    if (slot === 3) {                       // Year of Plenty — pick 2 (234..258)
      const anyLegal = [...legalDevIds].some(id => id >= 234 && id < 259);
      cards += anyLegal
        ? `<div class="dev-card" onclick="toggleDevPicker('yop')">${label}</div>`
        : `<div class="dev-card dev-dim" title="can't play now">${label}</div>`;
    }
  }
  // Inline pickers (hidden until the matching card is clicked).
  const monoBtns = [0,1,2,3,4].map(r =>
    legalDevIds.has(229 + r)
      ? `<button class="dev-pick" onclick="postAction(${229 + r})">${res(r)}</button>`
      : `<button class="dev-pick" disabled>${res(r)}</button>`).join(' ');
  const picker = `
    <div id="devPicker-mono" class="dev-picker" style="display:none">
      <span class="dev-pick-label">Monopolize:</span> ${monoBtns}</div>
    <div id="devPicker-yop" class="dev-picker" style="display:none">
      <span class="dev-pick-label">Year of Plenty — pick 2 resources:</span>
      <div id="yopIcons"></div>
      <span id="yopErr" class="dev-pick-err"></span></div>`;
  el.innerHTML = `<div class="trade-grid-label">Dev Cards</div>` +
                 `<div class="dev-hand">${cards}</div>` + picker;
  renderYopIcons(null);   // draw the icon picker fresh (1st-resource row)
}

// Year of Plenty via clickable resource icons. STATELESS: the in-progress
// first pick lives in the picker's `data-first` attribute (not a persistent
// global), so re-renders from state updates can't desync it. Same-kind picks
// (e.g. Wheat+Wheat, id 252) are fully supported — the engine allows all 25
// ordered pairs incl. r1==r2.
function yopLegalIds() {
  return new Set((G.state.legal_actions || [])
    .filter(a => a.id >= 234 && a.id < 259).map(a => a.id));
}

// Render the icon rows. `first` (0..4 or null) is the resource already picked.
function renderYopIcons(first) {
  const box = document.getElementById('yopIcons');
  if (!box) return;
  if (first === undefined) first = null;
  box.dataset.first = (first == null) ? '' : String(first);
  const legal = yopLegalIds();
  const validFirst  = r  => [0,1,2,3,4].some(r2 => legal.has(234 + r*5 + r2));
  const validSecond = r2 => first != null && legal.has(234 + first*5 + r2);
  let html = '';
  for (let r = 0; r < 5; r++) {
    const enabled = (first == null) ? validFirst(r) : validSecond(r);
    const sel = (first === r) ? ' yop-sel' : '';
    html += enabled
      ? `<button type="button" class="dev-pick yop-icon${sel}" onclick="pickYop(${r})">${res(r)}</button>`
      : `<button type="button" class="dev-pick yop-icon" disabled>${res(r)}</button>`;
  }
  const hint = (first == null) ? 'Click the 1st resource' : 'Now click the 2nd (same is OK)';
  html += (first != null)
    ? ` <button type="button" class="dev-pick" onclick="resetYop()" title="start over">↺</button>` : '';
  box.innerHTML = `<span class="yop-hint">${hint}</span> ${html}`;
}

function pickYop(r) {
  const box = document.getElementById('yopIcons');
  if (!box) return;
  const cur = box.dataset.first;
  if (cur === '' || cur === undefined) {
    // First pick — remember it in the DOM and redraw the 2nd-pick row.
    renderYopIcons(r);
    return;
  }
  const first = parseInt(cur, 10);
  const id = 234 + first * 5 + r;          // r may equal `first` (same-kind) — fine.
  const err = document.getElementById('yopErr');
  if (!yopLegalIds().has(id)) {
    if (err) err.textContent = 'that pair is not available';
    return;
  }
  if (err) err.textContent = '';
  box.dataset.first = '';                   // clear before the state refreshes
  postAction(id);
}

function resetYop() { renderYopIcons(null); }

// Show one picker at a time; clicking the same card again hides it.
function toggleDevPicker(which) {
  if (which === 'yop') { renderYopIcons(null); }  // fresh selection on open
  for (const w of ['mono', 'yop']) {
    const p = document.getElementById('devPicker-' + w);
    if (!p) continue;
    p.style.display = (w === which && p.style.display === 'none') ? 'block' : 'none';
  }
}

// Year of Plenty: post 234 + r1*5 + r2 iff that id is legal for this turn.
// Map each legal trade id to its [give,get] cell and render a 5×5 give→get
// matrix into containerId. Off-diagonal cells with no legal action are
// disabled. Shared by both the bank-trade and propose-trade grids.
function buildTradeGrid(containerId, ids, decodeFn, postFn) {
  const grid = document.getElementById(containerId);
  if (!grid) return;
  const cellId = {};   // "gi,gj" -> action id
  for (const id of ids) {
    const [gi, gj] = decodeFn(id);
    cellId[`${gi},${gj}`] = id;
  }
  let head = '<th class="tg-corner">give↓ get→</th>';
  for (let j = 0; j < 5; j++) head += `<th>${res(j)}</th>`;
  let rows = '';
  for (let i = 0; i < 5; i++) {
    let cells = `<th>${res(i)}</th>`;
    for (let j = 0; j < 5; j++) {
      if (i === j) { cells += `<td class="tg-diag"></td>`; continue; }
      const id = cellId[`${i},${j}`];
      if (id !== undefined) {
        cells += `<td><button class="tg-cell" onclick="${postFn.name}(${id})">↔</button></td>`;
      } else {
        cells += `<td><button class="tg-cell" disabled>·</button></td>`;
      }
    }
    rows += `<tr>${cells}</tr>`;
  }
  grid.innerHTML = `<table class="trade-grid"><tr>${head}</tr>${rows}</table>`;
}

function postTradeFromGrid(id) {
  postAction(id);
}

function clearTradeTimer() {
  if (G && G._tradeTimer) { clearInterval(G._tradeTimer); G._tradeTimer = null; }
}

async function respondTrade(accept) {
  clearTradeTimer();   // cancel auto-reject so the timer never fires post-close
  document.querySelectorAll('.modal-bg').forEach(m => m.remove());
  const r = await fetch(`/api/games/${G.gid}/trade-response`,
    { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ accept }) });
  applyState(await r.json());
}

function showTradeModal(o) {
  clearTradeTimer();   // a fresh offer gets a fresh 5s timer; kill any stale one
  document.querySelectorAll('.modal-bg').forEach(m => m.remove());
  const div = document.createElement('div');
  div.className = 'modal-bg';
  div.innerHTML = `<div class="modal">
    <p><b class="seat-${o.from_seat}">${G.state.seat_names[o.from_seat]}</b> offers a trade:</p>
    <p>You give ${res(o.you_give[0])}×${o.you_give[1]}, you get ${res(o.you_get[0])}×${o.you_get[1]}</p>
    <div class="trade-timer">Auto-reject in <span id="tradeCountdown">5</span>s
      <div class="trade-timer-bar"><div id="tradeTimerFill"></div></div></div>
    <button class="primary" onclick="respondTrade(true)">Accept</button>
    <button onclick="respondTrade(false)">Reject</button></div>`;
  document.body.appendChild(div);
  // 5-second countdown; at 0 auto-reject. Manual Accept/Reject clears it.
  let remaining = 5;
  const span = document.getElementById('tradeCountdown');
  const fill = document.getElementById('tradeTimerFill');
  if (fill) fill.style.width = '100%';
  G._tradeTimer = setInterval(() => {
    remaining -= 1;
    if (span) span.textContent = String(Math.max(remaining, 0));
    if (fill) fill.style.width = `${Math.max(remaining, 0) * 20}%`;
    if (remaining <= 0) { clearTradeTimer(); respondTrade(false); }
  }, 1000);
}

// ---- Victory celebration --------------------------------------------------
function celebrateGameOver(st) {
  let winner = -1;
  if (st.returns) winner = st.returns.indexOf(1);
  const humanWon = winner === st.human_seat;
  const winnerName = winner >= 0 ? st.seat_names[winner] : 'Nobody';
  // Banner / modal with a fresh-game button.
  const bg = document.createElement('div');
  bg.className = 'modal-bg win-modal-bg';
  const title = humanWon ? '🎉 You win!' : `${escapeHtml(winnerName)} wins`;
  bg.innerHTML = `<div class="modal win-modal">
    <h2 class="win-title">${title}</h2>
    <button class="primary" onclick="location.reload()">New Game</button>
  </div>`;
  document.body.appendChild(bg);
  // Full confetti for a human win; a subdued burst otherwise.
  fireConfetti(humanWon ? 150 : 40);
}

// Self-contained vanilla canvas confetti — no external library.
function fireConfetti(count) {
  const colors = [...PLAYER_COLORS, '#ffc107', '#ffd700'];
  const canvas = document.createElement('canvas');
  canvas.className = 'confetti-canvas';
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const pieces = [];
  for (let i = 0; i < count; i++) {
    pieces.push({
      x: Math.random() * canvas.width,
      y: -20 - Math.random() * canvas.height * 0.5,
      w: 6 + Math.random() * 8,
      h: 8 + Math.random() * 10,
      vy: 2 + Math.random() * 4,
      vx: -1.5 + Math.random() * 3,
      rot: Math.random() * Math.PI * 2,
      vr: -0.2 + Math.random() * 0.4,
      color: colors[(Math.random() * colors.length) | 0],
    });
  }
  const start = performance.now();
  const DURATION = 4000;
  function frame(now) {
    const elapsed = now - start;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of pieces) {
      p.x += p.vx; p.y += p.vy; p.rot += p.vr;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
      ctx.restore();
    }
    if (elapsed < DURATION) {
      requestAnimationFrame(frame);
    } else {
      canvas.remove();
    }
  }
  requestAnimationFrame(frame);
}

function maybeStreamBots(st) {
  // When bots are driving in the background, open SSE to get the settled state.
  if (st.status !== 'bot_thinking') return;
  if (G._sse) { G._sse.close(); G._sse = null; }
  G._sse = new EventSource(`/api/games/${G.gid}/events`);
  G._sse.onmessage = (ev) => {
    try {
      const next = JSON.parse(ev.data);
      applyStateNoStream(next);          // render without re-opening SSE
      if (next.status !== 'bot_thinking') { G._sse.close(); G._sse = null; }
    } catch (_) {}
  };
  G._sse.onerror = () => {
    if (G._sse) { G._sse.close(); G._sse = null; }
    // Stream closed (server SSE cap or network). Re-sync once; if bots are
    // still driving, re-open the stream so slow GNN/MCTS turns don't strand
    // the UI on "bot thinking".
    fetch(`/api/games/${G.gid}/state`).then(r => r.json()).then(st => {
      applyStateNoStream(st);
      // Backoff before re-opening so a proxy that fails SSE but serves REST
      // can't drive a tight reconnect loop.
      if (st.status === 'bot_thinking') setTimeout(() => maybeStreamBots(st), 500);
    }).catch(() => {});
  };
}
