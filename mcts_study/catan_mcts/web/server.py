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

    app.state.games = games
    app.state.checkpoints_dir = checkpoints_dir
    app.state.replays_dir = replays_dir
    return app
