"""Mirror a run's metrics log and text samples to Weights & Biases, so the
curves outlive the VM. A separate process that tails the files training
writes (<run_dir>/model.msgpack.metrics.jsonl and .samples.jsonl); training
itself never waits for it or depends on it.

The W&B run id is kept in <run_dir>/wandb_run_id, so a resumed run (same
VM or a new one) continues the same W&B run. Steps already sent are
skipped (<run_dir>/wandb_synced.json): after a crash, the steps training
repeats from its last checkpoint keep their first values in W&B, while the
local metrics log has both.

    WANDB_API_KEY=... python -m memory_pool_model.wandb_sync --run_dir runs/x --project memory-pool
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time

from .dashboard import JsonlTail, read_conf


def _numeric(prefix: str, rec: dict, skip=("step", "time", "type")) -> dict:
    out = {}
    for k, v in rec.items():
        if k in skip:
            continue
        if isinstance(v, bool):
            out[prefix + k] = int(v)
        elif isinstance(v, (int, float)):
            out[prefix + k] = v
        elif isinstance(v, dict) and k == "data":
            out[prefix + "data_tokens"] = v.get("tokens")
            out[prefix + "data_wait_s"] = v.get("wait_s")
    return out


class Syncer:
    def __init__(self, run_dir: str, wandb_mod, project: str, entity: str | None = None, name: str | None = None):
        self.run_dir, self.wandb = run_dir, wandb_mod
        base = os.path.join(run_dir, "model.msgpack")
        self.metrics = JsonlTail(base + ".metrics.jsonl")
        self.samples = JsonlTail(base + ".samples.jsonl")
        self.synced_path = os.path.join(run_dir, "wandb_synced.json")
        self.synced = {"step": 0, "sample_step": -1, "eval_step": -1}
        if os.path.exists(self.synced_path):
            with open(self.synced_path) as f:
                self.synced.update(json.load(f))
        id_path = os.path.join(run_dir, "wandb_run_id")
        if os.path.exists(id_path):
            with open(id_path) as f:
                run_id = f.read().strip()
        else:
            run_id = secrets.token_hex(4)  # W&B run ids: lowercase letters and digits
            with open(id_path, "w") as f:
                f.write(run_id)
        conf = read_conf(os.path.join(run_dir, "run.conf"))
        os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
        self.run = wandb_mod.init(project=project, entity=entity, name=name or os.path.basename(os.path.abspath(run_dir)),
                                  id=run_id, resume="allow", dir=run_dir, config={"run_conf": conf})
        self.step = self.synced["step"]  # last W&B step written
        self.finished = False
        self.expect_sample = -1  # step of the last sample snapshot started

    def _save(self) -> None:
        with open(self.synced_path + ".tmp", "w") as f:
            json.dump(self.synced, f)
        os.replace(self.synced_path + ".tmp", self.synced_path)

    def _log(self, data: dict, step: int) -> None:
        self.step = max(self.step, step)
        self.run.log(data, step=self.step)

    def poll(self) -> int:
        """Send what's new; returns the number of records sent."""
        sent = 0
        for r in self.metrics.new_records():
            t = r.get("type")
            if t == "train":
                if r["step"] <= self.synced["step"]:
                    continue
                self._log(_numeric("train/", r), r["step"])
                self.synced["step"] = r["step"]
            elif t == "eval":
                if r["step"] <= self.synced["eval_step"]:
                    continue
                self._log(_numeric("eval/", r), r["step"])
                self.synced["eval_step"] = r["step"]
            elif t == "checkpoint":
                self._log({"checkpoint/step": r["step"], "checkpoint/write_s": r.get("write_s"),
                           "checkpoint/waited_s": r.get("copy_s", 0) + r.get("waited_s", 0)}, r["step"])
            elif t == "start":
                self.run.config.update({"model": r.get("config", {}).get("model"),
                                        "train": r.get("config", {}).get("train"),
                                        "devices": r.get("devices"), "device_kind": r.get("device_kind"),
                                        "params": r.get("params")}, allow_val_change=True)
            elif t == "sample_started":
                self.expect_sample = max(self.expect_sample, r["step"])
                continue
            elif t == "end":
                self.run.summary["outcome"] = r.get("outcome")
                self.finished = r.get("outcome") == "finished"
            else:
                continue
            sent += 1
        for s in self.samples.new_records():
            if s["step"] <= self.synced["sample_step"]:
                continue
            table = self.wandb.Table(columns=["prompt", "text", "text_pool_shuffled", "mean_top_prob"])
            for x in s["samples"]:
                table.add_data(x["prompt"], x["text"], x.get("text_pool_shuffled"), x.get("mean_top_prob"))
            self._log({"samples/table": table, "samples/step": s["step"],
                       "samples/same_with_pool_shuffled": s.get("same_with_pool_shuffled")}, s["step"])
            self.synced["sample_step"] = s["step"]
            sent += 1
        if sent:
            self._save()
        return sent

    def close(self) -> None:
        self._save()
        self.run.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "memory-pool-lm"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args()
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") not in ("offline", "disabled"):
        print("WANDB_API_KEY is not set: not syncing to Weights & Biases", flush=True)
        return
    import wandb

    s = Syncer(a.run_dir, wandb, a.project, a.entity, a.name)
    print(f"syncing {a.run_dir} to W&B run {s.run.id} ({getattr(s.run, 'url', '')})", flush=True)
    try:
        while True:
            s.poll()
            if s.finished:
                # the last CPU sample can finish after training: wait for it (at most 15 min)
                deadline = time.time() + 900
                while s.synced["sample_step"] < s.expect_sample and time.time() < deadline:
                    time.sleep(a.poll)
                    s.poll()
                print("training finished; W&B sync done", flush=True)
                break
            time.sleep(a.poll)
    finally:
        s.close()


if __name__ == "__main__":
    main()
