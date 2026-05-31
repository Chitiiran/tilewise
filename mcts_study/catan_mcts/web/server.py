"""FastAPI app: serves the play-vs-bots frontend + REST/SSE API.

create_app(checkpoints_dir, replays_dir) returns a configured app. Paths are
parameters (no hardcoded WSL paths) so the same code runs locally or deployed.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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

    @app.get("/api/bots")
    def get_bots():
        return {
            "types": bot_registry.list_types(),
            "checkpoints": bot_registry.list_checkpoints(checkpoints_dir),
        }

    def _get(gid: str) -> GameSession:
        sess = games.get(gid)
        if sess is None:
            raise HTTPException(status_code=404, detail="game not found")
        return sess

    @app.post("/api/games")
    def create_game(spec: SetupSpec):
        try:
            sess = GameSession(spec.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        gid = uuid.uuid4().hex[:12]
        games[gid] = sess
        state = sess.advance()
        return {"game_id": gid, "board": sess.board_payload(), "state": state}

    @app.get("/api/games/{gid}/state")
    def get_state(gid: str):
        return _get(gid).state_json()

    @app.post("/api/games/{gid}/action")
    def post_action(gid: str, body: ActionBody):
        sess = _get(gid)
        try:
            return sess.apply_human_action(body.action)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/api/games/{gid}/trade-response")
    def post_trade(gid: str, body: TradeBody):
        return _get(gid).respond_to_trade(accept=body.accept)

    app.state.games = games
    app.state.checkpoints_dir = checkpoints_dir
    app.state.replays_dir = replays_dir
    return app
