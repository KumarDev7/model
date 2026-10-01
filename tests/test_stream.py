import json
import os
import threading
import time

import numpy as np
import pytest

from experiments import stream_ultrafineweb as su
from memory_pool_model.data import StreamingTokenDataset

L = 8  # window length - 1


def write_stream(d, shards, parts, done=True):
    """shards: {part: [n_tokens, ...]}; tokens are consecutive ids across all shards."""
    with open(os.path.join(d, "stream.json"), "w") as f:
        json.dump({"parts": parts, "vocab_size": 65536}, f)
    np.save(os.path.join(d, "val.npy"), np.arange(200, dtype=np.uint16))
    nxt = 0
    for p in parts:
        for k, n in enumerate(shards.get(p, [])):
            np.save(os.path.join(d, f"p{p:04d}-{k:04d}.npy"), np.arange(nxt, nxt + n, dtype=np.uint16))
            nxt += n
        if done:
            with open(os.path.join(d, f"p{p:04d}.done"), "w") as f:
                json.dump({"shards": len(shards.get(p, []))}, f)
    return nxt


def make_ds(d, **kw):
    kw = {"seq_len": L, "block_shards": 2, "poll_seconds": 0.02, **kw}
    return StreamingTokenDataset(str(d), os.path.join(d, "val.npy"), 65536, **kw)


def drain(ds, batch=5):
    out = []
    try:
        while True:
            out.append(ds.sample(None, batch))
    except RuntimeError:
        pass
    return out


def test_every_window_once_in_shuffled_order(tmp_path):
    write_stream(tmp_path, {5: [100, 90, 81], 7: [77]}, [5, 7])
    ds = make_ds(tmp_path, keep_consumed=True)
    batches = drain(ds)
    firsts = np.concatenate([b["inputs"][:, 0] for b in batches])
    for b in batches:  # windows are contiguous and targets are shifted inputs
        np.testing.assert_array_equal(b["inputs"] + 1, b["targets"])
        np.testing.assert_array_equal(np.diff(b["inputs"], axis=1), 1)
    expected = []
    off = 0
    for n in (100, 90, 81, 77):  # windows of L+1 tokens overlapping by one: every token is a target once
        expected += list(off + np.arange(0, n - L, L))
        off += n
    assert len(expected) == 42 and len(firsts) == 40  # 8 full batches of 5; the stream ends inside the 9th
    assert len(set(firsts.tolist())) == 40 and set(firsts.tolist()) <= set(expected)
    assert firsts.tolist() != sorted(firsts.tolist())  # shuffled within blocks
    blocks = [(0, 100 + 90), (190, 190 + 81 + 77)]  # block_shards=2; block 0 fully read before block 1
    assert all(blocks[0][0] <= x < blocks[0][1] for x in firsts[:20])


def test_resume_continues_exactly(tmp_path):
    write_stream(tmp_path, {5: [100, 90, 81], 7: [77]}, [5, 7])
    a = make_ds(tmp_path, keep_consumed=True)
    for _ in range(5):  # into block 1 (block 0 has 23 windows)
        a.sample(None, 5)
    st = json.loads(json.dumps(a.state_dict()))  # through the checkpoint json
    rest_a = drain(a)
    b = make_ds(tmp_path, keep_consumed=True)
    b.load_state_dict(st)
    rest_b = drain(b)
    assert len(rest_a) == len(rest_b) > 0
    for x, y in zip(rest_a, rest_b):
        np.testing.assert_array_equal(x["inputs"], y["inputs"])


def test_consumed_shards_deleted_only_after_checkpoint(tmp_path):
    write_stream(tmp_path, {5: [100, 90, 81], 7: [77]}, [5, 7])
    ds = make_ds(tmp_path, block_shards=1)
    ds.defer_delete = True
    files = lambda: sorted(f for f in os.listdir(tmp_path) if f.endswith(".npy") and f.startswith("p"))
    for _ in range(4):  # block 0 (11 windows) read, block 1 started
        ds.sample(None, 5)
    assert ds.state_dict()["block"] == ["p0005-0001.npy"]
    assert "p0005-0000.npy" in files()
    saved = ds.state_dict()  # a checkpoint taken now, written later in the background
    for _ in range(2):  # reading moves on into block 2 (3 windows left in block 1, then 7)
        ds.sample(None, 5)
    assert ds.state_dict()["block"] == ["p0005-0002.npy"]
    ds.checkpoint_saved(saved)  # deletes only what that checkpoint is past
    assert files() == ["p0005-0001.npy", "p0005-0002.npy", "p0007-0000.npy"]
    nd = make_ds(tmp_path, block_shards=1)  # without checkpoints: deleted as soon as a block is done
    nd.load_state_dict(ds.state_dict())
    drain(nd)
    assert files() == ["p0007-0000.npy"]


def test_waits_for_the_producer(tmp_path):
    write_stream(tmp_path, {5: [100]}, [5, 7], done=False)
    ds = make_ds(tmp_path, block_shards=1)

    def produce():
        time.sleep(0.3)
        with open(tmp_path / "p0005.done", "w") as f:
            json.dump({"shards": 1}, f)
        np.save(tmp_path / "p0007-0000.npy", np.arange(1000, 1100, dtype=np.uint16))
        with open(tmp_path / "p0007.done", "w") as f:
            json.dump({"shards": 1}, f)

    threading.Thread(target=produce).start()
    batches = drain(ds)
    firsts = np.concatenate([b["inputs"][:, 0] for b in batches])
    assert len(firsts) == 20 and (firsts >= 1000).sum() == 8 and ds.wait_seconds > 0  # 24 windows, 4 batches


def test_missing_shard_of_finished_part_is_an_error(tmp_path):
    write_stream(tmp_path, {5: [100, 90]}, [5])
    os.remove(tmp_path / "p0005-0000.npy")
    with pytest.raises(FileNotFoundError):
        make_ds(tmp_path).sample(None, 5)


# ------------------------------------------------------------------ producer
class _Enc:
    def __init__(self, ids):
        self.ids = ids


class FakeTok:
    def token_to_id(self, t):
        return 0

    def encode_batch(self, texts):
        return [_Enc([ord(c) for c in t]) for t in texts]


def rows(n, fail_at=None):
    for i in range(n):
        if i == fail_at:
            raise ConnectionError("simulated network error")
        yield "" if i % 7 == 3 else f"doc {i} " + "x" * (i % 13)


def shard_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".npy"))


def test_producer_restart_gives_identical_shards(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    kw = dict(tok=FakeTok(), parts=[1], shard_tokens=200, max_ready=10**9, chunk_docs=4)
    n = su.produce_part(str(a), 1, rows=rows(300), **kw)
    with pytest.raises(ConnectionError):
        su.produce_part(str(b), 1, rows=rows(300, fail_at=150), **kw)
    assert not os.path.exists(b / "p0001.done") and os.path.exists(b / "p0001.progress")
    assert su.produce_part(str(b), 1, rows=rows(300), **kw) == n > 3
    assert shard_files(a) == shard_files(b)
    for f in shard_files(a):
        np.testing.assert_array_equal(np.load(a / f), np.load(b / f))
    whole = np.concatenate([np.load(a / f) for f in shard_files(a)])
    assert (whole == 0).sum() == sum(1 for t in rows(300) if t)  # one end-of-text per non-empty document
    assert su.produce_part(str(a), 1, rows=iter(()), **kw) == n  # a finished part isn't read again


def test_producer_waits_for_disk_but_not_for_the_oldest_part(tmp_path):
    kw = dict(tok=FakeTok(), parts=[1, 3], shard_tokens=200, max_ready=0, chunk_docs=4, poll=0.02)
    later = threading.Thread(target=su.produce_part, args=(str(tmp_path), 3), kwargs=dict(rows=rows(100), **kw))
    later.start()  # part 3 is ahead of unfinished part 1 and the disk budget is 0: it must wait
    time.sleep(0.3)
    assert not any(f.startswith("p0003-") for f in os.listdir(tmp_path))
    su.produce_part(str(tmp_path), 1, rows=rows(100), **kw)  # the oldest part writes despite the budget
    assert os.path.exists(tmp_path / "p0001.done")
    later.join(5)  # now part 3 is the oldest unfinished part
    assert not later.is_alive() and os.path.exists(tmp_path / "p0003.done")


def test_new_vm_resume_from_checkpoint(tmp_path, monkeypatch):
    """Empty stream dir + training checkpoint in part 3: parts 1 and 3's earlier shards aren't needed,
    part 1 is skipped, part 3 is regenerated identically and the reader continues."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(), new.mkdir()
    kw = dict(tok=FakeTok(), parts=[1, 3], shard_tokens=200, max_ready=10**9, chunk_docs=4)
    for d in (old,):
        for p in (1, 3):
            su.produce_part(str(d), p, rows=rows(100 + p), **kw)
    with open(old / "stream.json", "w") as f:
        json.dump({"parts": [1, 3], "vocab_size": 65536}, f)
    np.save(old / "val.npy", np.arange(200, dtype=np.uint16))
    a = make_ds(old, block_shards=1, keep_consumed=True)
    while a.state_dict()["part"] == 0 or a.state_dict()["block"][0].startswith("p0001"):
        a.sample(None, 5)
    st = a.state_dict()
    expect = drain(a)
    ckpt = tmp_path / "run.state.json"
    ckpt.write_text(json.dumps({"step": 7, "data_state": st}))
    monkeypatch.setattr(su, "hf_rows", lambda part: rows(100 + part))
    monkeypatch.setattr(su.sys, "argv", ["x", "--out", str(new), "--tokenizer", "unused", "--parts", "1,3",
                                         "--shard_tokens", "200", "--from_checkpoint", str(ckpt)])
    jobs = []
    monkeypatch.setattr(su, "_run_jobs", lambda js, workers: jobs.extend(js))
    monkeypatch.setattr(su, "_load_tokenizer", lambda path: (b"{}", 65536))
    su.main()
    assert json.loads((new / "p0001.done").read_text())["skipped"] and [j[1] for j in jobs] == [3]
    su.produce_part(str(new), 3, rows=rows(103), **kw)
    np.save(new / "val.npy", np.arange(200, dtype=np.uint16))
    b = make_ds(new, block_shards=1, keep_consumed=True)
    b.load_state_dict(st)
    got = drain(b)
    assert len(got) == len(expect) > 0
    for x, y in zip(got, expect):
        np.testing.assert_array_equal(x["inputs"], y["inputs"])


def test_parse_parts():
    assert su.parse_parts("1,3-6,9") == [1, 3, 4, 5, 6, 9]
