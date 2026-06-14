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
        from catan_az import analytics
        ladder = _read_json(loop_root / "ladder.json", {})
        status = _read_json(loop_root / "status.json", {})
        journal = _read_csv(loop_root / "journal.csv")
        champ = (ladder.get("entries", {}) or {}).get(ladder.get("champion"), {})
        return {
            "champion": champ,
            "status": status,
            "journal": journal[-10:],
            "holds_since_promotion": analytics.holds_since_promotion(loop_root),
            "flags": analytics.detect_failure_modes(loop_root),
            "play_champion_url": f"http://localhost:{web_port}/?difficulty=az-champion",
        }

    @app.get("/api/metrics")
    def metrics():
        """Per-iteration derived metrics (winrate/draw/timeout/Elo trends)."""
        from catan_az import analytics
        return {"iterations": analytics.iteration_metrics(loop_root)}

    @app.get("/api/seat-bias/{iter_n}")
    def seat_bias(iter_n: int):
        """Per-seat candidate winrate for an iteration's arena (board-luck check)."""
        from catan_az import analytics
        return analytics.seat_bias(loop_root, iter_n=iter_n)

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
