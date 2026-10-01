"""Live web dashboard of a training run: curves, eval, generated text, the
data stream, checkpoints, processes, disk and logs.

It only reads the files a run writes (scripts/train_stream.sh layout):

    <run_dir>/run.conf                          settings (STEPS, STREAM_DIR, ...)
    <run_dir>/model.msgpack.metrics.jsonl       one JSON line per step + events
    <run_dir>/model.msgpack.samples.jsonl       text generated on the CPU
    <run_dir>/model.msgpack.state.json          last checkpoint
    <run_dir>/*.pid, <run_dir>/logs/*.log       processes and their logs

so it never touches the training process. Every request needs the access
token (?token=... in the page URL), because the page is reachable from the
internet through a tunnel.

    python -m memory_pool_model.dashboard --run_dir runs/x --port 8765
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import secrets
import shutil
import threading
import time
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

import numpy as np

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
PROCESSES = {  # pid file -> text its command line contains
    "supervisor": ("supervisor.pid", "train_stream"),
    "training": ("train.pid", "memory_pool_model.train"),
    "stream producer": ("stream.pid", "stream_ultrafineweb"),
    "wandb sync": ("wandb.pid", "memory_pool_model.wandb_sync"),
    "tunnel": ("tunnel.pid", "cloudflared"),
}


def read_conf(path: str) -> Dict[str, str]:
    conf = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                m = re.match(r'^([A-Z_]+)=(.*)$', line.strip())
                if m:
                    conf[m.group(1)] = m.group(2).strip().strip('"')
    return conf


def count_parts(spec: str) -> int | None:
    """"1,3-2047" -> 2046"""
    try:
        return sum(int(hi or lo) - int(lo) + 1 for lo, _, hi in (x.partition("-") for x in spec.split(",") if x))
    except ValueError:
        return None


def proc_running(pid_file: str, pattern: str) -> bool:
    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return pattern in f.read().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError):
        return False


def tail(path: str, n: int = 80, max_bytes: int = 256_000) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        lines = f.read().decode(errors="replace").splitlines()
    return lines[-n:]


class JsonlTail:
    """Reads new complete lines of a growing JSONL file."""

    def __init__(self, path: str):
        self.path, self.offset, self._buf = path, 0, b""

    def new_records(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        size = os.path.getsize(self.path)
        if size < self.offset:  # truncated / replaced: start over
            self.offset, self._buf = 0, b""
        if size == self.offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        self.offset = size
        data = self._buf + data
        lines = data.split(b"\n")
        self._buf = lines.pop()  # incomplete last line
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out


class RunState:
    """In-memory columns of the run, refreshed from the files."""

    def __init__(self, run_dir: str, save_name: str = "model.msgpack"):
        self.run_dir = run_dir
        self.base = os.path.join(run_dir, save_name)
        self.metrics = JsonlTail(self.base + ".metrics.jsonl")
        self.samples_tail = JsonlTail(self.base + ".samples.jsonl")
        self.lock = threading.Lock()
        self.steps = array("d")
        self.cols: Dict[str, array] = {}
        self.last: dict = {}
        self.events: Dict[str, List[dict]] = {}
        self.samples: List[dict] = []
        self.warnings: List[dict] = []

    # -------------------------------------------------------------- columns
    def _truncate(self, step: float) -> None:
        """A resumed run repeats steps after its checkpoint: drop the old ones."""
        i = int(np.searchsorted(np.frombuffer(self.steps, dtype="d"), step))
        del self.steps[i:]
        for c in self.cols.values():
            del c[i:]
        self.warnings = [w for w in self.warnings if w["step"] < step]

    def _add_train(self, r: dict) -> None:
        step = float(r["step"])
        if len(self.steps) and step <= self.steps[-1]:
            self._truncate(step)
        n = len(self.steps)
        self.steps.append(step)
        for k, v in r.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or k in ("step", "time"):
                continue
            c = self.cols.get(k)
            if c is None:
                c = self.cols[k] = array("d", [float("nan")] * n)
            c.append(float(v))
        for c in self.cols.values():  # fields missing from this record
            if len(c) < n + 1:
                c.append(float("nan"))
        if r.get("nonfinite"):
            self.warnings.append({"step": r["step"], "kind": "non-finite loss / gradient"})
        if "grad_spike" in r:
            top = sorted(((k[3:], v) for k, v in r.items() if k.startswith("gn_") and isinstance(v, (int, float))),
                         key=lambda kv: -kv[1])[:3]
            self.warnings.append({"step": r["step"], "kind": f"grad spike {r['grad_spike']:.1f}x",
                                  "detail": ", ".join(f"{k}={v:.3g}" for k, v in top)})
        self.last = r

    def refresh(self) -> None:
        with self.lock:
            for r in self.metrics.new_records():
                t = r.get("type")
                if t == "train":
                    self._add_train(r)
                else:
                    lst = self.events.setdefault(t, [])
                    lst.append(r)
                    del lst[:-500]
            for r in self.samples_tail.new_records():
                self.samples.append(r)
                del self.samples[:-50]

    # ------------------------------------------------------------------ API
    def series(self, fields: List[str], points: int) -> dict:
        with self.lock:
            steps = np.frombuffer(self.steps, dtype="d").copy()
            cols = {f: np.frombuffer(self.cols[f], dtype="d").copy() for f in fields if f in self.cols}
        n = len(steps)
        out = {"n": n, "fields": {}}
        if n == 0:
            out["step"] = []
            return out
        b = max(1, -(-n // points))
        m = n // b * b
        rest = n - m

        def agg(x, fn):
            parts = []
            if m:
                with np.errstate(all="ignore"), _quiet():
                    parts.append(fn(x[:m].reshape(-1, b), axis=1))
            if rest:
                with np.errstate(all="ignore"), _quiet():
                    parts.append(np.array([fn(x[m:])]))
            return np.concatenate(parts)

        out["step"] = agg(steps, np.max).tolist()
        for f, x in cols.items():
            out["fields"][f] = {"mean": _jsonable(agg(x, np.nanmean)), "max": _jsonable(agg(x, np.nanmax)),
                                "min": _jsonable(agg(x, np.nanmin))}
        return out

    def fields(self) -> List[str]:
        with self.lock:
            return sorted(self.cols)

    def status(self) -> dict:
        conf = read_conf(os.path.join(self.run_dir, "run.conf"))
        with self.lock:
            last = dict(self.last)
            steps = np.frombuffer(self.steps, dtype="d").copy()  # a view would block appends
            recent = {k: np.frombuffer(self.cols[k], dtype="d")[-200:].copy() for k in
                      ("step_s", "loss", "tokens_per_s", "mfu", "pause_s") if k in self.cols}
            ev = {k: v[-1] for k, v in self.events.items() if v}
            evals = list(self.events.get("eval", []))
            ckpts = list(self.events.get("checkpoint", []))[-20:]
            warnings = list(self.warnings)[-50:]
            starts = list(self.events.get("start", []))
        total = int(conf.get("STEPS", 0) or 0) or (starts[-1]["config"]["train"]["steps"] if starts else 0)
        step = int(last.get("step", 0))
        with np.errstate(all="ignore"), _quiet():
            step_s = float(np.nanmedian(recent["step_s"])) if "step_s" in recent and len(recent["step_s"]) else None
            summ = {k: (float(np.nanmean(v[-100:])) if len(v) else None) for k, v in recent.items()}
        busy = 0.0
        if "step_s" in self.cols:
            with self.lock:
                st_all = np.frombuffer(self.cols["step_s"], dtype="d").copy()
                gaps = np.diff(np.concatenate([[steps[0] - 1], steps])) if len(steps) else steps
            busy = float(np.nansum(st_all[: len(gaps)] * gaps))
        paused = float(np.nansum(np.frombuffer(self.cols["pause_s"], dtype="d"))) if "pause_s" in self.cols else 0.0
        procs = {name: proc_running(os.path.join(self.run_dir, f), pat) for name, (f, pat) in PROCESSES.items()}
        stream_dir = conf.get("STREAM_DIR", "")
        stream = {}
        if stream_dir and os.path.isdir(stream_dir):
            shards = glob.glob(os.path.join(stream_dir, "p????-????.npy"))
            done = 0
            for p in glob.glob(os.path.join(stream_dir, "p????.done")):
                try:
                    with open(p) as f:
                        done += not json.load(f).get("skipped", False)
                except (OSError, json.JSONDecodeError):
                    pass
            stream = {"dir": stream_dir, "shards_ready": len(shards),
                      "gb_ready": sum(os.path.getsize(s) for s in shards if os.path.exists(s)) / 1e9,
                      "parts_done": done, "parts_total": count_parts(conf.get("PARTS", ""))}
        disk = {}
        for label, path in (("run", self.run_dir), ("stream", stream_dir)):
            if path and os.path.exists(path):
                u = shutil.disk_usage(path)
                disk[label] = {"free_gb": u.free / 1e9, "total_gb": u.total / 1e9}
        ckpt_step = None
        try:
            with open(self.base + ".state.json") as f:
                ckpt_step = json.load(f)["step"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        url = None
        try:
            with open(os.path.join(self.run_dir, "dashboard_url.txt")) as f:
                url = f.read().strip().split("?")[0]
        except OSError:
            pass
        return {
            "run": os.path.basename(os.path.abspath(self.run_dir)), "now": time.time(),
            "step": step, "total_steps": total, "last_record_time": last.get("time"),
            "eta_s": (total - step) * step_s if step_s and total else None, "step_s": step_s,
            "recent": summ, "last": {k: v for k, v in last.items() if not isinstance(v, dict)},
            "data": last.get("data"), "paused_share": paused / max(busy + paused, 1e-9),
            "checkpoint_step": ckpt_step, "checkpoints": ckpts, "evals": evals[-200:], "warnings": warnings,
            "events": {k: v for k, v in ev.items() if k in ("start", "end", "sample_started", "sample_done",
                                                             "sample_skipped")},
            "sessions": len(starts), "start": starts[-1] if starts else None,
            "processes": procs, "stream": stream, "disk": disk, "conf": conf, "public_url": url,
        }

    def get_samples(self) -> List[dict]:
        with self.lock:
            return list(self.samples)

    def logs(self, name: str) -> List[str]:
        files = {"train": "logs/train.log", "stream": "logs/stream.log", "supervisor": "logs/supervisor.log",
                 "samples": "model.msgpack.samples.log", "wandb": "logs/wandb.log", "tunnel": "logs/tunnel.log"}
        if name not in files:
            return []
        return tail(os.path.join(self.run_dir, files[name]), 150)


class _quiet:
    """Silence 'mean of empty slice' warnings for all-NaN buckets."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def _jsonable(x: np.ndarray) -> list:
    return [None if not np.isfinite(v) else round(float(v), 7) for v in x]


def make_handler(state: RunState, token: str):
    page = open(PAGE, "rb").read()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            q = parse_qs(u.query)
            given = (q.get("token") or [self.headers.get("X-Token", "")])[0]
            if not secrets.compare_digest(given, token):
                return self._send(403, b"forbidden: open the URL with ?token=...", "text/plain")
            if u.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            try:
                state.refresh()
                if u.path == "/api/status":
                    data = state.status()
                elif u.path == "/api/series":
                    fields = [f for f in (q.get("fields") or [""])[0].split(",") if f]
                    data = state.series(fields, min(int((q.get("points") or ["1500"])[0]), 5000))
                elif u.path == "/api/fields":
                    data = state.fields()
                elif u.path == "/api/samples":
                    data = state.get_samples()
                elif u.path == "/api/logs":
                    data = state.logs((q.get("name") or ["train"])[0])
                else:
                    return self._send(404, b"not found", "text/plain")
            except Exception as e:  # keep serving; show the error on the page
                return self._send(500, json.dumps({"error": repr(e)}).encode())
            return self._send(200, json.dumps(data, allow_nan=False, default=str).encode())

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="the tunnel connects locally")
    ap.add_argument("--token", default=os.environ.get("DASHBOARD_TOKEN"),
                    help="access token (default: <run_dir>/dashboard.token, created once)")
    a = ap.parse_args()
    token = a.token
    tok_path = os.path.join(a.run_dir, "dashboard.token")
    if not token:
        if os.path.exists(tok_path):
            with open(tok_path) as f:
                token = f.read().strip()
        else:
            token = secrets.token_urlsafe(18)
            os.makedirs(a.run_dir, exist_ok=True)
            fd = os.open(tok_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(token)
    state = RunState(a.run_dir)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(state, token))
    print(f"dashboard: http://{a.host}:{a.port}/?token={token}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
