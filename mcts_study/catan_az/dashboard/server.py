"""Minimal AZ progress dashboard: one JSON summary endpoint + static page.
Reads journal.csv / status.json / ladder.json (no DB). Spec 2026-06-13 §7."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_STATIC = Path(__file__).parent / "static"


def create_dashboard(*, loop_root, web_port: int = 8000) -> FastAPI:
    loop_root = Path(loop_root)
    app = FastAPI(title="AZ Daily Dashboard")

    @app.get("/api/summary")
    def summary():
        ladder = _read_json(loop_root / "ladder.json", {})
        status = _read_json(loop_root / "status.json", {})
        journal = _read_csv(loop_root / "journal.csv")
        champ = (ladder.get("entries", {}) or {}).get(ladder.get("champion"), {})
        return {
            "champion": champ,
            "status": status,
            "journal": journal[-10:],
            # deep-link into the existing web app's lobby, az-champion tier
            "play_champion_url": f"http://localhost:{web_port}/?difficulty=az-champion",
        }

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _read_csv(p: Path):
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))
