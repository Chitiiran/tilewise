"""FastAPI app: serves the play-vs-bots frontend + REST/SSE API.

create_app(checkpoints_dir, replays_dir) returns a configured app. Paths are
parameters (no hardcoded WSL paths) so the same code runs locally or deployed.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from catan_mcts.web import bot_registry
from catan_mcts.web.game_session import GameSession

_STATIC = Path(__file__).parent / "static"


class SetupSpec(BaseModel):
    human_seat: int
    seats: dict
    rules: dict | None = None
    seed: int | None = None


class ActionBody(BaseModel):
    action: int


class TradeBody(BaseModel):
    accept: bool


def create_app(*, checkpoints_dir, replays_dir) -> FastAPI:
    app = FastAPI(title="Catan Play-vs-Bots")
    checkpoints_dir = Path(checkpoints_dir)
    replays_dir = Path(replays_dir)
    games: dict[str, GameSession] = {}

    # Local-dev convenience: tell the browser never to cache the frontend assets
    # (index.html / play.js / replay.js / style.css) so edits show up on a plain
    # reload — no hard-refresh needed. The game/replay HTML is small, so the
    # bandwidth cost is negligible. (Drop this for a real CDN deployment.)
    @app.middleware("http")
    async def _no_store_frontend(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static") or path.startswith("/replays"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/api/bots")
    def get_bots():
        return {
            "types": bot_registry.list_types(),
            "difficulties": bot_registry.list_difficulties(),
            "checkpoints": bot_registry.list_checkpoints(checkpoints_dir),
        }

    def _get(gid: str) -> GameSession:
        sess = games.get(gid)
        if sess is None:
            raise HTTPException(status_code=404, detail="game not found")
        return sess

    @app.post("/api/games")
    def create_game(spec: SetupSpec):
        setup = spec.model_dump()
        try:
            # Expand {"difficulty": ...} seats into full bot specs before the
            # session sees them; explicit {"type": ...} specs pass through.
            setup["seats"] = {
                seat: bot_registry.resolve_seat_spec(
                    s, checkpoints_dir=checkpoints_dir)
                for seat, s in setup["seats"].items()
            }
            sess = GameSession(setup)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        gid = uuid.uuid4().hex[:12]
        games[gid] = sess
        state = sess.start_async()
        return {"game_id": gid, "board": sess.board_payload(), "state": state}

    @app.get("/api/games/{gid}/state")
    def get_state(gid: str):
        return _get(gid).state_json()

    @app.post("/api/games/{gid}/action")
    def post_action(gid: str, body: ActionBody):
        sess = _get(gid)
        try:
            return sess.apply_human_action_async(body.action)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/api/games/{gid}/trade-response")
    def post_trade(gid: str, body: TradeBody):
        sess = _get(gid)
        try:
            return sess.respond_to_trade_async(accept=body.accept)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/api/games/{gid}/events")
    def events(gid: str):
        sess = _get(gid)

        def gen():
            snap = sess.state_json()
            yield f"data: {json.dumps(snap)}\n\n"
            last = snap["status"]
            # Drive-completion stream: while bots run in the background, emit
            # whenever the status transitions; always emit the final settled
            # state once advancing stops. (Status-transition contract: a long
            # run of same-status bot moves yields no intermediate events.)
            # NOTE: state_json() acquires the session lock, which advance()
            # holds for the entire drive. During a long bot turn this poll
            # blocks inside state_json() rather than cycling, so this
            # 1200x50ms bound is NOT a hard ~60s wall-clock cap mid-turn — the
            # client-side SSE reconnect (play.js) is what keeps the UI live.
            for _ in range(1200):
                advancing = sess.is_advancing()
                cur = sess.state_json()
                if cur["status"] != last:
                    yield f"data: {json.dumps(cur)}\n\n"
                    last = cur["status"]
                if not advancing:
                    break
                time.sleep(0.05)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/")
    def index():
        idx = _STATIC / "index.html"
        if not idx.exists():
            raise HTTPException(status_code=404, detail="frontend not built")
        return FileResponse(idx)

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/api/replays")
    def list_replays():
        out = []
        if replays_dir.exists():
            for d in sorted(replays_dir.glob("playback_seed_*")):
                if (d / "index.html").exists():
                    out.append({"name": d.name, "url": f"/replays/{d.name}/index.html"})
        return {"replays": out}

    if replays_dir.exists():
        app.mount("/replays", StaticFiles(directory=str(replays_dir)), name="replays")

    app.state.games = games
    app.state.checkpoints_dir = checkpoints_dir
    app.state.replays_dir = replays_dir
    return app
