#!/usr/bin/env python3
"""HAMC live dashboard.

Spins up a tiny HTTP server that streams the generation events of the three
HAMC layers (global, intermediate, local) over Server-Sent Events so a browser
can animate the wave-function-collapse process in real time.

Usage:
    python dashboard.py                       # serve on http://127.0.0.1:8765
    python dashboard.py --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from hamc.config.tile_config import TileConfig
from hamc.generators.global_generator import GlobalGenerator
from hamc.generators.intermediate_generator import IntermediateGenerator
from hamc.generators.local_generator import LocalGenerator

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_static")


# --- Resource limits (DoS / abuse hardening) ---------------------------------
# A single GET /api/stream spawns a full generation. Without caps a handful of
# EventSource connections could pin every core and grow memory without bound.
MAX_CONCURRENT_RUNS = 3          # generations running at once, server-wide
MAX_TOTAL_CELLS = 128 * 128      # budget on final tilemap area per request
MAX_QUEUE_SIZE = 2000            # bounded broker queue (back-pressure)
MAX_CLIENT_LOG_BYTES = 16 * 1024 # cap on POST /api/client-log body

# Server-wide gate on concurrent generations. ``acquire(blocking=False)`` lets
# us reject excess work immediately instead of queueing unbounded threads.
_run_gate = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)


class EventBroker:
    """Per-run pub/sub. One generation run feeds one SSE client.

    The queue is bounded so a slow or vanished consumer applies back-pressure
    instead of letting the producer buffer events without limit. ``emit`` is
    cancellation-aware: once the client disconnects (``close``/``done``) the
    producer stops enqueuing and can bail out of the generation loop.
    """

    def __init__(self) -> None:
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.done = threading.Event()

    def emit(self, event: Dict[str, Any]) -> None:
        if self.done.is_set():
            return
        try:
            self.queue.put(event, timeout=5.0)
        except queue.Full:
            # Consumer is gone or hopelessly behind: stop the run.
            self.done.set()

    def cancelled(self) -> bool:
        return self.done.is_set()

    def close(self) -> None:
        self.done.set()
        try:
            self.queue.put_nowait({"type": "_eof"})
        except queue.Full:
            pass


def _local_origin(origin: Optional[str]) -> Optional[str]:
    """Return ``origin`` only if it is a loopback origin, else ``None``.

    Reflecting an arbitrary ``Access-Control-Allow-Origin: *`` let any website
    drive the local generator (and, with ``--host 0.0.0.0``, reach it over the
    LAN). We only echo CORS for loopback origins so a random page can't read
    the stream cross-origin.
    """
    if not origin:
        return None
    try:
        host = urlparse(origin).hostname
    except ValueError:
        return None
    if host in ("127.0.0.1", "localhost", "::1"):
        return origin
    return None


def _sanitize_log(value: Any, limit: int = 2000) -> str:
    """Flatten newlines and clamp length so client logs can't forge entries."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text[:limit]


REGION_COLORS = {
    "forest": (52, 138, 78),
    "desert": (224, 198, 142),
    "city":   (132, 138, 152),
}

BLOCK_COLORS = {
    "grove":       (33, 110, 50),
    "clearing":    (124, 198, 73),
    "river":       (76, 175, 230),
    "sand":        (224, 198, 142),
    "oasis":       (60, 158, 130),
    "residential": (172, 70, 56),
    "commercial":  (170, 210, 230),
    "industrial":  (170, 175, 182),
    "road":        (62, 62, 70),
    "scrubland":   (154, 195, 90),
    "periurban":   (148, 152, 158),
    "village":     (180, 132, 80),
}


def _palette() -> Dict[str, Any]:
    return {
        "regions": {k: list(v) for k, v in REGION_COLORS.items()},
        "blocks":  {k: list(v) for k, v in BLOCK_COLORS.items()},
        "tiles":   {k: list(v) for k, v in TileConfig.COLORS.items()},
    }


PARAM_LIMITS = {
    "width":   (1, 8),
    "height":  (1, 8),
    "subgrid": (1, 4),
    "local":   (2, 8),
}


def _clip(name: str, raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    lo, hi = PARAM_LIMITS[name]
    return max(lo, min(hi, value))


def _run_generation(params: Dict[str, Any], broker: EventBroker) -> None:
    """Run a full HAMC generation, streaming events through the broker."""
    width = _clip("width",   params.get("width"),   4)
    height = _clip("height",  params.get("height"),  3)
    subgrid = _clip("subgrid", params.get("subgrid"), 2)
    local_size = _clip("local",   params.get("local"),   4)
    try:
        delay = max(0.0, min(1.0, float(params.get("delay", 0.04))))
    except (TypeError, ValueError):
        delay = 0.04
    seed = params.get("seed")

    # Reject runs whose final tilemap would exceed the cost budget before doing
    # any work. Final area = (height*subgrid*local) * (width*subgrid*local).
    final_h = height * subgrid * local_size
    final_w = width * subgrid * local_size
    if final_h * final_w > MAX_TOTAL_CELLS:
        broker.emit({
            "type": "run_error",
            "error": (
                f"Requested map ({final_w}x{final_h} = {final_w * final_h} tiles) "
                f"exceeds the budget of {MAX_TOTAL_CELLS} tiles."
            ),
        })
        broker.close()
        return

    # Cap concurrent generations server-wide. If we're at capacity, fail fast
    # rather than spawning yet another CPU-bound thread.
    if not _run_gate.acquire(blocking=False):
        broker.emit({
            "type": "run_error",
            "error": "Server busy: too many concurrent generations. Try again shortly.",
        })
        broker.close()
        return

    # Save module-level RNG state so we can restore it after this run.
    # Without this, two concurrent SSE streams using ThreadingHTTPServer
    # would corrupt each other's random state (M-7 in the audit).
    _prev_random_state = random.getstate()
    _prev_np_state = np.random.get_state()

    if seed not in (None, ""):
        try:
            seed_int = int(seed)
            random.seed(seed_int)
            np.random.seed(seed_int)
        except (TypeError, ValueError):
            pass

    start_t = time.time()
    broker.emit({
        "type": "run_start",
        "params": {
            "width": width, "height": height,
            "subgrid": subgrid, "local": local_size, "delay": delay,
        },
        "palette": _palette(),
    })

    gen_logger = logging.getLogger("hamc.gen")

    def tag(level: str):
        def relay(event: Dict[str, Any]) -> None:
            event["level"] = level
            event["t"] = time.time() - start_t
            etype = event.get("type")
            if etype == "fail":
                gen_logger.error(
                    "[%s] fail reason=%s generator=%s",
                    level, event.get("reason"), event.get("generator"),
                )
            elif etype == "conflict":
                gen_logger.warning(
                    "[%s] conflict at %s value=%s",
                    level, event.get("pos"), event.get("value"),
                )
            elif etype == "backtrack":
                gen_logger.info("[%s] backtrack at %s", level, event.get("pos"))
            broker.emit(event)
        return relay

    timings: Dict[str, float] = {}

    try:
        # ---- Global ----
        t0 = time.time()
        G = GlobalGenerator(width, height)
        G.set_observer(tag("global"), step_delay=delay)
        G.initialize()
        G.collapse()
        regions = [[c.collapsed_value for c in row] for row in G.cells]
        timings["global"] = time.time() - t0

        # ---- Intermediate ----
        t0 = time.time()
        I = IntermediateGenerator(G, subgrid_size=subgrid)
        I.set_observer(tag("intermediate"), step_delay=delay * 0.6)
        I.collapse()
        blocks = [[c.collapsed_value for c in row] for row in I.cells]
        timings["intermediate"] = time.time() - t0

        # ---- Local: stitched tilemap, one block at a time ----
        t0 = time.time()
        int_h, int_w = len(I.cells), len(I.cells[0])
        final_h, final_w = int_h * local_size, int_w * local_size
        tilemap: List[List[Optional[str]]] = [
            [None for _ in range(final_w)] for _ in range(final_h)
        ]
        broker.emit({
            "type": "local_init",
            "level": "local",
            "height": final_h, "width": final_w,
            "block_size": local_size,
            "t": time.time() - start_t,
        })

        for br in range(int_h):
            if broker.cancelled():
                return
            for bc in range(int_w):
                if broker.cancelled():
                    return
                btype = I.cells[br][bc].collapsed_value or "clearing"
                neighbours = _block_neighbours(I, br, bc)
                broker.emit({
                    "type": "local_block_start",
                    "level": "local",
                    "br": br, "bc": bc,
                    "block_type": btype,
                    "size": local_size,
                    "t": time.time() - start_t,
                })

                base_r, base_c = br * local_size, bc * local_size

                def local_relay(event: Dict[str, Any],
                                base_r=base_r, base_c=base_c, br=br, bc=bc) -> None:
                    event["level"] = "local"
                    event["br"] = br
                    event["bc"] = bc
                    event["base"] = [base_r, base_c]
                    event["t"] = time.time() - start_t
                    broker.emit(event)

                LG = LocalGenerator(btype, local_size, neighbours)
                LG.set_observer(local_relay, step_delay=delay * 0.25)
                LG.collapse()

                local_tiles = [[c.collapsed_value for c in row] for row in LG.cells]
                for r in range(local_size):
                    for c in range(local_size):
                        tilemap[base_r + r][base_c + c] = local_tiles[r][c]

                broker.emit({
                    "type": "local_block_complete",
                    "level": "local",
                    "br": br, "bc": bc,
                    "tiles": local_tiles,
                    "t": time.time() - start_t,
                })

        timings["local"] = time.time() - t0
        broker.emit({
            "type": "run_complete",
            "elapsed": time.time() - start_t,
            "timings": timings,
            "regions": regions,
            "blocks": blocks,
            "tilemap": tilemap,
        })

    except Exception as e:  # pragma: no cover - surfaced to the UI
        logging.getLogger("hamc.gen").exception("Generation failed: %s", e)
        broker.emit({"type": "run_error", "error": f"{type(e).__name__}: {e}"})
    finally:
        random.setstate(_prev_random_state)
        np.random.set_state(_prev_np_state)
        _run_gate.release()
        broker.close()


def _block_neighbours(I: IntermediateGenerator, r: int, c: int) -> Dict[str, str]:
    h, w = len(I.cells), len(I.cells[0])
    out: Dict[str, str] = {}
    def _val(rr: int, cc: int) -> Optional[str]:
        return I.cells[rr][cc].collapsed_value
    if r > 0 and (v := _val(r - 1, c)):
        out["top"] = v
    if r < h - 1 and (v := _val(r + 1, c)):
        out["bottom"] = v
    if c > 0 and (v := _val(r, c - 1)):
        out["left"] = v
    if c < w - 1 and (v := _val(r, c + 1)):
        out["right"] = v
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "HAMC-dashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logging.getLogger("dashboard").debug("%s - " + format, self.address_string(), *args)

    def _send_cors(self) -> None:
        """Echo CORS only for loopback origins (see ``_local_origin``)."""
        origin = _local_origin(self.headers.get("Origin"))
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/client-log":
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            # Cap the body so a client can't make us reserve arbitrary memory.
            length = max(0, min(length, MAX_CLIENT_LOG_BYTES))
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("client log payload must be an object")
            except Exception:
                payload = {"level": "error", "msg": "<unparseable client log>"}
            level = _sanitize_log(payload.get("level") or "error", 16).lower()
            msg = _sanitize_log(payload.get("msg") or "<no message>")
            extras = []
            for key in ("source", "line", "col", "url"):
                v = payload.get(key)
                if v not in (None, ""):
                    extras.append(f"{key}={_sanitize_log(v, 256)}")
            stack = payload.get("stack")
            client_logger = logging.getLogger("hamc.client")
            line = f"[client] {msg}" + (f" ({', '.join(extras)})" if extras else "")
            # Sanitized level may not name a real log method; fall back to error.
            log_fn = getattr(client_logger, level, client_logger.error)
            if not callable(log_fn):
                log_fn = client_logger.error
            log_fn(line)
            if stack:
                client_logger.error("[client] stack: %s", _sanitize_log(stack, 4000))
            self.send_response(204)
            self._send_cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_error(404, f"Not found: {parsed.path}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        routes = {
            "/":          ("index.html", "text/html; charset=utf-8"),
            "/index.html":("index.html", "text/html; charset=utf-8"),
            "/app.js":    ("app.js", "application/javascript; charset=utf-8"),
            "/app.css":   ("app.css", "text/css; charset=utf-8"),
        }
        if path in routes:
            name, ctype = routes[path]
            return self._serve_static(name, ctype)
        if path == "/api/health":
            return self._serve_json({"ok": True})
        if path == "/api/stream":
            return self._serve_stream(parse_qs(parsed.query))
        self.send_error(404, f"Not found: {parsed.path}")

    def _serve_json(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, name: str, content_type: str) -> None:
        path = os.path.join(STATIC_DIR, name)
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404, f"Missing asset: {name}")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self, qs: Dict[str, list]) -> None:
        params = {k: v[0] for k, v in qs.items()}
        broker = EventBroker()
        thread = threading.Thread(
            target=_run_generation, args=(params, broker), daemon=True,
        )
        thread.start()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        # CORS is echoed only for loopback origins so a foreign web page can't
        # open an EventSource against us and read the stream.
        self._send_cors()
        self.end_headers()

        # Send a retry hint up front so EventSource won't fight the server.
        self.wfile.write(b"retry: 1000\n\n")
        self.wfile.flush()

        try:
            while True:
                try:
                    event = broker.queue.get(timeout=15.0)
                except queue.Empty:
                    # Keepalive comment to defeat intermediate buffering.
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if event.get("type") == "_eof":
                    break
                payload = "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            broker.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAMC live dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open browser on start")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.warning(
            "Binding to %s exposes the generator beyond loopback. Anyone who can "
            "reach this host can trigger generations (CPU/memory cost). Prefer "
            "--host 127.0.0.1 unless you intend LAN access.",
            args.host,
        )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    logging.info("HAMC dashboard listening on %s", url)
    logging.info("  UI:     %s/", url)
    logging.info("  Stream: %s/api/stream?width=4&height=3&subgrid=2&local=4", url)
    logging.info("  Health: %s/api/health", url)
    if args.open:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
