"""Dashboard server with SSE push.

Drop-in replacement for `python -m http.server 8765` that:
  - Serves the static dashboard files (HTML, JS, JSON, PNGs).
  - Watches `runs/v3/dashboard/grid_*.json` for mtime changes.
  - Pushes `event: update` to all connected SSE clients on every change.

Usage:
    python scripts/dashboard_server.py [--port 8765] [--watch-dir runs/v3/dashboard]

The browser-side change is in grid_dashboard.html: replace `setInterval(refresh, REFRESH_MS)`
with an EventSource listener on /events. Clients automatically reconnect on errors.
"""
from __future__ import annotations
import argparse
import json
import queue
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

WATCH_FILES = ["grid_dashboard.json", "grid_full20.json", "grid_pass3.json", "grid_pass3_lastepoch.json", "grid_pass100k.json"]
WATCH_INTERVAL_S = 1.0
SSE_KEEPALIVE_S = 25.0


class Hub:
    """Fan-out broadcaster. Each SSE client gets its own queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[str]] = []

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=64)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def broadcast(self, payload: str) -> None:
        with self._lock:
            for q in list(self._clients):
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def n_clients(self) -> int:
        with self._lock:
            return len(self._clients)


def watcher_loop(hub: Hub, watch_dir: Path) -> None:
    """Poll mtime of WATCH_FILES; on any change, broadcast a JSON event."""
    last: dict[str, float] = {}
    while True:
        changed: list[str] = []
        for name in WATCH_FILES:
            p = watch_dir / name
            try:
                m = p.stat().st_mtime
            except FileNotFoundError:
                continue
            if last.get(name) != m:
                last[name] = m
                changed.append(name)
        if changed:
            hub.broadcast(json.dumps({"changed": changed, "ts": time.time()}))
        time.sleep(WATCH_INTERVAL_S)


def make_handler(hub: Hub, base: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(base), **kw)

        def log_message(self, fmt, *args):
            # Quiet — don't spam stdout for every poll.
            return

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/events":
                self._serve_sse()
                return
            super().do_GET()

        def _serve_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = hub.subscribe()
            try:
                # Send an immediate hello so client knows it's connected.
                self._write_event("hello", json.dumps({"clients": hub.n_clients()}))
                last_keepalive = time.time()
                while True:
                    try:
                        payload = q.get(timeout=1.0)
                        self._write_event("update", payload)
                    except queue.Empty:
                        pass
                    if time.time() - last_keepalive > SSE_KEEPALIVE_S:
                        # Comment lines keep the connection alive through proxies.
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last_keepalive = time.time()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

        def _write_event(self, event: str, data: str) -> None:
            try:
                self.wfile.write(f"event: {event}\n".encode())
                for line in data.splitlines() or [""]:
                    self.wfile.write(f"data: {line}\n".encode())
                self.wfile.write(b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

    return Handler


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--root", type=Path, default=Path.cwd(),
                   help="Document root to serve (default: cwd)")
    p.add_argument("--watch-dir", type=Path, default=Path("runs/v3/dashboard"),
                   help="Directory containing grid_*.json files to watch")
    args = p.parse_args()

    hub = Hub()
    watch_dir = args.watch_dir if args.watch_dir.is_absolute() else (args.root / args.watch_dir)
    t = threading.Thread(target=watcher_loop, args=(hub, watch_dir), daemon=True)
    t.start()

    handler = make_handler(hub, args.root)
    srv = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Dashboard server: http://{args.bind}:{args.port}/")
    print(f"  doc root:  {args.root}")
    print(f"  watching:  {watch_dir} ({', '.join(WATCH_FILES)})")
    print(f"  SSE endpoint: /events")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
