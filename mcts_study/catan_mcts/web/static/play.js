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
