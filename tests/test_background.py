import json
import os
import time

import jax
import numpy as np
import pytest

from memory_pool_model.background import AsyncCheckpointer, BackgroundJob, JsonlLog
from memory_pool_model.config import ModelConfig, TrainConfig
from memory_pool_model.data import FactDataset, TokenDataset
from memory_pool_model.train import run


def small(**kw):
    ds = FactDataset(num_entities=64, num_relations=2, num_attributes=16, name_alphabet=8, name_len=2, facts_per_seq=4)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4)
    tcfg = TrainConfig(**{"steps": 12, "batch_size": 16, "warmup_steps": 2, "revive_every": 5, "log_every": 5,
                          "eval_every": 6, **kw})
    return ds, mcfg, tcfg


def records(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_metrics_log_has_every_step_and_event(tmp_path):
    ds, mcfg, tcfg = small(checkpoint_every=4)
    run(mcfg, tcfg, ds, save_path=str(tmp_path / "m"))
    recs = records(tmp_path / "m.metrics.jsonl")
    kinds = [r["type"] for r in recs]
    assert kinds[0] == "start" and kinds[-1] == "end" and recs[-1]["outcome"] == "finished"
    train = [r for r in recs if r["type"] == "train"]
    assert [r["step"] for r in train] == list(range(1, 13))  # one line per step, in order
    for r in train:
        for k in ("loss", "ce", "acc", "grad_norm", "lr", "param_norm", "update_norm", "logit_max", "step_s",
                  "tokens_per_s", "tokens", "data_s", "nonfinite", "gn_attn", "gn_layer0", "gn_pool_values"):
            assert k in r, k
        assert r["nonfinite"] is False and r["tokens"] == r["step"] * 16 * (ds.seq_len - 1)
    assert train[0]["first"] and train[5]["lr"] > train[0]["lr"]  # warmup
    assert [r["step"] for r in recs if r["type"] == "eval"] == [6, 12]
    ck = [r for r in recs if r["type"] == "checkpoint"]
    assert [r["step"] for r in ck] == [4, 8, 12, 12] and ck[-1]["sync"]  # async every 4; final one in step
    assert all(r["write_s"] >= 0 for r in ck)


def test_async_checkpoints_resume_exactly(tmp_path):
    ds, mcfg, tcfg = small(steps=20, checkpoint_every=3, eval_every=100, log_every=100)
    _, full, _ = run(mcfg, tcfg, ds, save_path=str(tmp_path / "a"))
    run(mcfg, tcfg, ds, save_path=str(tmp_path / "b"), stop_after=10)
    _, resumed, _ = run(mcfg, tcfg, ds, save_path=str(tmp_path / "b"), resume=True)
    for a, b in zip(jax.tree_util.tree_leaves(full), jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with open(tmp_path / "b.state.json") as f:
        assert json.load(f)["step"] == 20


def test_checkpoint_write_does_not_block_and_errors_surface(tmp_path):
    ck = AsyncCheckpointer(JsonlLog(str(tmp_path / "log.jsonl")))
    state = {"w": jax.numpy.ones((4, 4))}
    written = []

    def slow_write(host_state):
        time.sleep(0.5)
        written.append(host_state)

    t = time.perf_counter()
    r = ck.save("x", state, slow_write, step=1)
    assert time.perf_counter() - t < 0.3 and isinstance(r["host_state"]["w"], np.ndarray)
    assert ck.wait() > 0.1 and len(written) == 1
    ck.save("x", state, lambda hs: 1 / 0, step=2)
    with pytest.raises(RuntimeError, match="background checkpoint failed"):
        ck.wait()
    assert [r["step"] for r in records(tmp_path / "log.jsonl")] == [1]


def test_background_job_runs_one_at_a_time():
    job, order = BackgroundJob("t"), []
    job.start(lambda: (time.sleep(0.2), order.append(1)))
    job.start(lambda: order.append(2))  # waits for the first
    job.wait()
    assert order == [1, 2]


def test_nonfinite_steps_are_flagged(tmp_path):
    ds, mcfg, tcfg = small(steps=3, lr=1e30, warmup_steps=1, eval_every=100)
    run(mcfg, tcfg, ds, save_path=str(tmp_path / "m"))
    train = [r for r in records(tmp_path / "m.metrics.jsonl") if r["type"] == "train"]
    assert any(r["nonfinite"] for r in train)


def _tiny_tokenizer(path):
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.BPE())
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"],
                                             initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["the cat sat on the mat. the dog ran in the park."] * 50, trainer=trainer)
    tok.save(str(path))
    return tok


def test_samples_generated_on_cpu_while_training(tmp_path):
    tok = _tiny_tokenizer(tmp_path / "tokenizer.json")
    V = tok.get_vocab_size()
    ids = np.asarray(tok.encode("the cat sat on the mat. the dog ran in the park. " * 400).ids, np.uint16)
    np.save(tmp_path / "train.npy", ids)
    np.save(tmp_path / "val.npy", ids[:2000])
    ds = TokenDataset(str(tmp_path / "train.npy"), str(tmp_path / "val.npy"), V, seq_len=32, eval_windows=16)
    mcfg = ModelConfig(vocab_size=V, max_len=32, d_model=32, n_heads=2, n_layers=2, memory_layers=(1,),
                       n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4)
    tcfg = TrainConfig(steps=40, batch_size=16, warmup_steps=2, log_every=20, eval_every=40, sample_every=20,
                       sample_tokens=6, sample_cpus=1, nopool_true_coef=0.0)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("the cat\nthe dog ran\n")
    save = str(tmp_path / "m")
    run(mcfg, tcfg, ds, save_path=save, sample={"tokenizer": str(tmp_path / "tokenizer.json"), "prompts": str(prompts)})
    out = tmp_path / "m.samples.jsonl"
    deadline = time.time() + 240
    while time.time() < deadline and (not out.exists() or len(out.read_text().splitlines()) < 1):
        time.sleep(1)
    log = (tmp_path / "m.samples.log").read_text() if (tmp_path / "m.samples.log").exists() else ""
    assert out.exists(), log
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert recs[0]["step"] in (20, 40) and [s["prompt"] for s in recs[0]["samples"]] == ["the cat", "the dog ran"]
    assert all(isinstance(s["text"], str) and "text_pool_shuffled" in s for s in recs[0]["samples"])
    events = [r for r in records(tmp_path / "m.metrics.jsonl") if r["type"].startswith("sample")]
    assert events[0]["type"] == "sample_started" and events[0]["step"] == 20
