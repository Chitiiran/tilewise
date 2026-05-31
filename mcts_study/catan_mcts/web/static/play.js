const PLAY = document.getElementById('tab-play');
let BOTS = null;

async function initLobby() {
  BOTS = await (await fetch('/api/bots')).json();
  renderLobby();
}

function botSelect(seat) {
  const opts = BOTS.types.map(t => `<option value="${t.id}">${t.label}</option>`).join('');
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
        &nbsp; seed <input id="seed" type="number" placeholder="random" style="width:90px">
      </div>
      <button class="primary" id="start">Start Game</button>
      <span id="lobby-err" style="color:#c33"></span>
    </div>`;
  syncHumanSeat();
  PLAY.querySelectorAll('input[name=human]').forEach(r => r.onchange = syncHumanSeat);
  PLAY.querySelectorAll('.bot-type').forEach(sel => sel.onchange = () => syncCkpt(sel));
  document.getElementById('start').onclick = onStart;
}

function syncHumanSeat() {
  const human = +PLAY.querySelector('input[name=human]:checked').value;
  for (let s = 0; s < 4; s++) {
    const cell = PLAY.querySelector(`.seat-bot[data-seat="${s}"]`);
    cell.style.visibility = (s === human) ? 'hidden' : 'visible';
  }
}

function syncCkpt(sel) {
  const seat = sel.dataset.seat;
  const ck = PLAY.querySelector(`.bot-ckpt[data-seat="${seat}"]`);
  const type = BOTS.types.find(t => t.id === sel.value);
  if (type && type.needs_checkpoint) {
    ck.innerHTML = BOTS.checkpoints.map(c => `<option value="${c.path}">${c.name}</option>`).join('')
                   || '<option value="">(no .pt found)</option>';
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
  PLAY.innerHTML = `
    <div class="row">
      <div class="panel board-col">
        <div id="boardWrap">
          <img id="board" src="data:image/png;base64,${G.png}">
          <svg id="overlay" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
        <div id="actionBar" style="margin-top:8px"></div>
      </div>
      <div class="panel" style="flex:1 1 360px; min-width:340px">
        <div id="status"></div>
        <div id="players"></div>
        <h3 style="font-size:13px;margin:8px 0 4px">Log</h3>
        <div id="log"></div>
      </div>
    </div>`;
  document.getElementById('board').addEventListener('load', renderGame);
  applyState(G.state);
}

const PLAYER_COLORS = ["#cc3333", "#3366cc", "#33aa55", "#cc8833"];
const RES = ['🪵','🧱','🐑','🌾','⛰️'];
// Letter fallback for environments without a color-emoji font (mirrors
// board_layout.RESOURCE_LETTER). Wrapped so a missing glyph still reads.
const RES_LETTER = ['W', 'B', 'S', 'Wh', 'O'];
function res(r) { return `<span title="${RES_LETTER[r]}">${RES[r]}</span>`; }

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
  if (st.status === 'trade_offer') showTradeModal(st.trade_offer);
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
  // Settlements + cities.
  for (const [vid,o] of st.s) {
    const v = G.layout.vertices[String(vid)]; const [px,py] = dataToPx(v[0],v[1]);
    body += `<rect x="${px-8}" y="${py-8}" width="16" height="16" fill="${PLAYER_COLORS[o]}" stroke="#222" stroke-width="1.5"/>`;
  }
  for (const [vid,o] of st.c) {
    const v = G.layout.vertices[String(vid)]; const [px,py] = dataToPx(v[0],v[1]);
    body += `<rect x="${px-10}" y="${py-10}" width="20" height="20" rx="3" fill="${PLAYER_COLORS[o]}" stroke="#fff" stroke-width="2"/>`;
  }
  // Clickable spatial targets when it's your turn (Task 19 fills click handlers).
  body += spatialTargetsSvg(G.state);
  svg.innerHTML = body;
  renderPlayers(G.state);
  renderStatus(G.state);
}

function renderPlayers(g) {
  const st = g.state; let rows = '';
  for (let i = 0; i < 4; i++) {
    const h = st.hands[i];
    const hand = h.breakdown.map((n,r) => n>0?`${res(r)}${n}`:'').filter(Boolean).join(' ');
    const me = i === g.human_seat ? ' (You)' : '';
    const cp = g.current_player === i ? '▶ ' : '';
    rows += `<tr><td class="seat-${i}"><b>${cp}${g.seat_names[i]}${me}</b></td>
             <td>${st.vp[i]} VP</td><td>${hand||'—'}</td></tr>`;
  }
  document.getElementById('players').innerHTML =
    `<table><tr><th>seat</th><th>VP</th><th>hand</th></tr>${rows}</table>`;
}

function renderStatus(g) {
  const map = { your_turn: 'Your turn', bot_thinking: 'Bot thinking…',
                trade_offer: 'Trade offer', game_over: 'Game over', error: 'Error' };
  let txt = map[g.status] || g.status;
  if (g.status === 'game_over' && g.returns) {
    const w = g.returns.indexOf(1);
    txt = (w === g.human_seat) ? 'You win 🎉' : `${g.seat_names[w]} wins`;
  }
  document.getElementById('status').innerHTML = `<b>${txt}</b>`;
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
    let px, py;
    if (a.kind === 'build_road') {
      const e = G.layout.edges[a.target]; [px,py] = dataToPx((e[0]+e[2])/2,(e[1]+e[3])/2);
    } else if (a.kind === 'move_robber') {
      const c = G.layout.hex_centers[a.target]; [px,py] = dataToPx(c[0],c[1]);
    } else {
      const v = G.layout.vertices[String(a.target)]; [px,py] = dataToPx(v[0],v[1]);
    }
    out += `<circle class="clickable" cx="${px}" cy="${py}" r="10" fill="#ffd633" fill-opacity="0.5"
             stroke="#c90" stroke-width="2" style="pointer-events:all" onclick="postAction(${a.id})"/>`;
  }
  return out;
}

function renderActionBar(g) {
  const bar = document.getElementById('actionBar');
  if (g.status !== 'your_turn' || !g.legal_actions) { bar.innerHTML = ''; return; }
  // Non-spatial actions become buttons; spatial ones are board clicks.
  const NON_SPATIAL = new Set(['roll','end_turn','buy_dev','trade_bank','propose_trade','play_dev','discard']);
  const seen = new Map();
  for (const a of g.legal_actions) {
    if (!NON_SPATIAL.has(a.kind)) continue;
    if (!seen.has(a.id)) seen.set(a.id, a);
  }
  bar.innerHTML = [...seen.values()]
    .map(a => `<button onclick="postAction(${a.id})">${a.label}</button>`).join(' ');
}

async function respondTrade(accept) {
  document.querySelectorAll('.modal-bg').forEach(m => m.remove());
  const r = await fetch(`/api/games/${G.gid}/trade-response`,
    { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ accept }) });
  applyState(await r.json());
}

function showTradeModal(o) {
  document.querySelectorAll('.modal-bg').forEach(m => m.remove());
  const div = document.createElement('div');
  div.className = 'modal-bg';
  div.innerHTML = `<div class="modal">
    <p><b class="seat-${o.from_seat}">${G.state.seat_names[o.from_seat]}</b> offers a trade:</p>
    <p>You give ${res(o.you_give[0])}×${o.you_give[1]}, you get ${res(o.you_get[0])}×${o.you_get[1]}</p>
    <button class="primary" onclick="respondTrade(true)">Accept</button>
    <button onclick="respondTrade(false)">Reject</button></div>`;
  document.body.appendChild(div);
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
