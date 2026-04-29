"""Serve the Spark.js viewer + the per-run outputs as a single static site.

Layout served at the chosen port:

    /                  -> repo/viewer/index.html
    /index.html        -> repo/viewer/index.html
    /main.js           -> repo/viewer/main.js
    /runs/<run_id>/... -> repo/outputs/runs/<run_id>/...   (the .ply lives here)

Open `http://127.0.0.1:<port>/index.html?run=<run_id>` to load a specific run,
or just `http://127.0.0.1:<port>/index.html` to use the latest run.
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import latest_run_dir, load_config  # noqa: E402


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Maps `/runs/<rid>/...` to the absolute outputs path; everything else
    is served from the viewer directory."""

    viewer_dir: Path
    runs_dir: Path

    def translate_path(self, path: str) -> str:  # type: ignore[override]
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/runs/"):
            rel = clean[len("/runs/"):]
            return str((self.runs_dir / rel).resolve())
        rel = clean.lstrip("/")
        if rel == "":
            rel = "index.html"
        return str((self.viewer_dir / rel).resolve())

    def end_headers(self):  # type: ignore[override]
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-id", default="", help="Specific run id; empty -> latest")
    p.add_argument("--port", type=int, default=0, help="0 -> use config viewer.port")
    p.add_argument("--bind", default="", help="empty -> use config viewer.bind")
    args = p.parse_args()

    cfg = load_config(args.config)
    port = args.port or int(cfg.viewer.port)
    bind = args.bind or str(cfg.viewer.bind)

    viewer_dir = REPO_ROOT / "viewer"
    runs_dir = REPO_ROOT / cfg.run.outputs_dir if not Path(cfg.run.outputs_dir).is_absolute() else Path(cfg.run.outputs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    rid = args.run_id or str(cfg.viewer.default_run_id) or ""
    if not rid:
        latest = latest_run_dir(cfg)
        if latest is not None:
            rid = latest.name

    handler_cls = type(
        "BoundHandler",
        (ViewerHandler,),
        {"viewer_dir": viewer_dir, "runs_dir": runs_dir},
    )
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((bind, port), handler_cls) as httpd:
        suffix = f"?run={rid}" if rid else ""
        print(f"[serve_viewer] http://{bind}:{port}/index.html{suffix}", flush=True)
        print(f"[serve_viewer] viewer_dir={viewer_dir}", flush=True)
        print(f"[serve_viewer] runs_dir  ={runs_dir}", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve_viewer] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
