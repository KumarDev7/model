"""Work that runs next to training without holding up the accelerator: the
metrics log, checkpoint writing, and text samples generated on the CPU.

Training only waits for the device-to-host copy of the arrays (the step
that produced them has to finish anyway); serialising and writing happen in
a background thread, generation in a separate low-priority process.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict

import jax
import numpy as np
from flax import serialization

# Dense bf16 peak per chip, for the MFU estimate (None: unknown device)
PEAK_FLOPS = {
    "TPU v4": 275e12,
    "TPU v5 lite": 197e12,
    "TPU v5e": 197e12,
    "TPU v5": 459e12,  # v5p
    "TPU v5p": 459e12,
    "TPU v6 lite": 918e12,
    "TPU v6e": 918e12,
    "NVIDIA A100-SXM4-40GB": 312e12,
    "NVIDIA A100-SXM4-80GB": 312e12,
    "NVIDIA H100 80GB HBM3": 989e12,
    "Tesla T4": 65e12,
}


def peak_flops(device) -> float | None:
    return PEAK_FLOPS.get(getattr(device, "device_kind", ""))


def device_memory() -> Dict[str, float]:
    """HBM in use / peak on the first local device, in GB (empty on CPU)."""
    try:
        s = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # some backends don't report memory
        return {}
    out = {}
    if "bytes_in_use" in s:
        out["hbm_gb"] = s["bytes_in_use"] / 1e9
    if "peak_bytes_in_use" in s:
        out["hbm_peak_gb"] = s["peak_bytes_in_use"] / 1e9
    if "bytes_limit" in s:
        out["hbm_limit_gb"] = s["bytes_limit"] / 1e9
    return out


def host_rss_gb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def _clean(v):
    """JSON-safe floats (NaN/inf become strings so the line stays valid JSON)."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


class JsonlLog:
    """One JSON object per line, appended; safe to write from several threads.
    Every record has "type" and "time" (unix seconds)."""

    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()
        self._f = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._f = open(path, "a", buffering=1)

    def write(self, rec: Dict[str, Any]) -> None:
        if self._f is None:
            return
        line = json.dumps(_clean({"type": rec.get("type", "train"), "time": round(time.time(), 3), **rec}))
        with self._lock:
            self._f.write(line + "\n")

    def close(self) -> None:
        if self._f is not None:
            with self._lock:
                self._f.close()
                self._f = None


class BackgroundJob:
    """At most one job at a time in a background thread. An exception in the
    job is raised again in the training thread at the next wait()."""

    def __init__(self, name: str):
        self.name = name
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self) -> float:
        """Block until the running job is done; seconds waited."""
        t0 = time.perf_counter()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError(f"background {self.name} failed") from err
        return time.perf_counter() - t0

    def start(self, fn: Callable[[], None]) -> None:
        self.wait()

        def target():
            try:
                fn()
            except BaseException as e:  # reported by the next wait()
                self._error = e

        # not a daemon: the interpreter finishes a write before exiting
        self._thread = threading.Thread(target=target, name=self.name, daemon=False)
        self._thread.start()


def write_atomic(path: str, data: bytes) -> None:
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)


class AsyncCheckpointer:
    """save() copies the train state to host memory and returns; a background
    thread serialises and writes it (temp file + rename, so a crash leaves the
    previous checkpoint intact). on_done(record) runs after the write."""

    def __init__(self, log: JsonlLog | None = None):
        self.job = BackgroundJob("checkpoint")
        self.log = log

    def save(self, path: str, state, write: Callable[[Any], None], step: int,
             on_done: Callable[[], None] | None = None) -> Dict[str, float]:
        """write(host_state) writes the files. Returns the seconds training
        waited: for the previous write to finish and for the device copy."""
        waited = self.job.wait()
        t0 = time.perf_counter()
        host_state = jax.device_get(state)
        copy_s = time.perf_counter() - t0
        nbytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(host_state) if hasattr(x, "nbytes"))

        def work():
            t1 = time.perf_counter()
            write(host_state)
            if on_done is not None:
                on_done()
            if self.log is not None:
                self.log.write({"type": "checkpoint", "step": step, "path": path, "gb": nbytes / 1e9,
                                "write_s": time.perf_counter() - t1, "copy_s": copy_s, "waited_s": waited})

        self.job.start(work)
        return {"waited_s": waited, "copy_s": copy_s, "host_state": host_state}

    def wait(self) -> float:
        return self.job.wait()


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Q: What is the boiling point of water at sea level? A:",
    "Photosynthesis is the process by which",
    "The best way to learn a new language is",
    "In 1969, Neil Armstrong",
    "Here is a simple recipe for pancakes:",
    "The main difference between a virus and a bacterium is",
    "Once upon a time, in a small village,",
]


class SampleLauncher:
    """Every sample_every steps: copy the parameters to host memory, write them
    to <save>.sample_params.msgpack in a background thread and start
    memory_pool_model.sample_worker on the CPU (low priority, its own cores).
    The worker appends one JSON line per snapshot to <save>.samples.jsonl.
    A snapshot is skipped while the previous one is still generating."""

    def __init__(self, save_path: str, tokenizer: str, prompts: str | None = None, cpus: int = 0,
                 new_tokens: int = 48, log: JsonlLog | None = None):
        self.save_path, self.tokenizer, self.prompts = save_path, tokenizer, prompts
        self.cpus, self.new_tokens, self.log = cpus, new_tokens, log
        self.params_path = save_path + ".sample_params.msgpack"
        self.out_path = save_path + ".samples.jsonl"
        self.job = BackgroundJob("sample snapshot")
        self.proc: subprocess.Popen | None = None
        self._proc_step = None
        self._proc_t0 = 0.0

    def running(self) -> bool:
        return self.busy_writing() or (self.proc is not None and self.proc.poll() is None)

    def busy_writing(self) -> bool:
        return self.job.busy()

    def poll(self) -> None:
        """Log a finished worker."""
        if self.proc is not None and self.proc.poll() is not None:
            if self.log is not None:
                self.log.write({"type": "sample_done", "step": self._proc_step, "returncode": self.proc.returncode,
                                "seconds": time.time() - self._proc_t0})
            if self.proc.returncode != 0:
                print(f"  sample worker for step {self._proc_step} failed (exit {self.proc.returncode}), "
                      f"see {self.save_path}.samples.log", flush=True)
            self.proc = None

    def launch(self, step: int, params=None, host_params=None) -> Dict[str, Any]:
        """Start a snapshot. Pass host_params when a host copy already exists
        (e.g. from a checkpoint at the same step). Returns what happened."""
        self.poll()
        if self.running():
            if self.log is not None:
                self.log.write({"type": "sample_skipped", "step": step, "reason": "previous sample still running"})
            return {"skipped": True}
        t0 = time.perf_counter()
        if host_params is None:
            host_params = jax.device_get(params)
        copy_s = time.perf_counter() - t0

        def work():
            write_atomic(self.params_path, serialization.to_bytes(host_params))
            cmd = [sys.executable, "-m", "memory_pool_model.sample_worker", "--params", self.params_path,
                   "--config", self.save_path + ".config.json", "--tokenizer", self.tokenizer,
                   "--step", str(step), "--out", self.out_path, "--new_tokens", str(self.new_tokens),
                   "--cpus", str(self.cpus)]
            if self.prompts:
                cmd += ["--prompts", self.prompts]
            env = {**os.environ, "JAX_PLATFORMS": "cpu"}
            pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
            with open(self.save_path + ".samples.log", "a") as logf:
                # no preexec_fn: it would fork() this multithreaded process (the worker lowers its own priority)
                self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            self._proc_step, self._proc_t0 = step, time.time()

        self.job.start(work)
        if self.log is not None:
            self.log.write({"type": "sample_started", "step": step, "copy_s": copy_s})
        return {"skipped": False, "copy_s": copy_s}

    def wait_written(self) -> None:
        self.job.wait()
