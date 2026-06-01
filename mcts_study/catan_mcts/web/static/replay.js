const REPLAY = document.getElementById('tab-replay');

async function initReplay() {
  const data = await (await fetch('/api/replays')).json();
  if (!data.replays.length) {
    REPLAY.innerHTML = `<div class="panel">No replays found. Generate one with
      <code>python -m catan_mcts.playback &lt;run_dir&gt; &lt;seed&gt;</code> into the
      server's <code>--replays-dir</code>.</div>`;
    return;
  }
  REPLAY.innerHTML = `<div class="panel"><h2>Replays</h2><ul>` +
    data.replays.map(r => `<li><a href="${r.url}" target="_blank">${r.name}</a></li>`).join('') +
    `</ul></div>`;
}

initReplay();
