import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from memory_pool_model.dashboard import RunState, count_parts, make_handler


def write(path, recs, mode="a"):
    with open(path, mode) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def train(step, loss, **kw):
    return {"type": "train", "step": step, "time": 1.0 + step, "loss": loss, "step_s": 0.1, "pause_s": 0.0, **kw}


def test_columns_follow_a_resume_and_partial_lines(tmp_path):
    m = tmp_path / "model.msgpack.metrics.jsonl"
    write(m, [{"type": "start", "step": 0}] + [train(s, 10.0 - s) for s in range(1, 9)])
    st = RunState(str(tmp_path))
    st.refresh()
    assert list(st.steps) == list(range(1, 9))
    # resumed from a checkpoint at step 5: steps 6.. are logged again with new values
    write(m, [{"type": "start", "step": 5}] + [train(s, 100.0 + s, grad_spike=7.0, gn_attn=3.0) for s in (6, 7)])
    with open(m, "a") as f:
        f.write('{"type": "train", "step": 8, "lo')  # a line still being written
    st.refresh()
    assert list(st.steps) == [1, 2, 3, 4, 5, 6, 7] and list(st.cols["loss"])[-2:] == [106.0, 107.0]
    assert [w["step"] for w in st.warnings] == [6, 7] and "attn" in st.warnings[0]["detail"]
    with open(m, "a") as f:
        f.write('ss": 1.5}\n')
    st.refresh()
    assert st.steps[-1] == 8 and st.cols["loss"][-1] == 1.5
    assert len(st.cols["gn_attn"]) == len(st.steps)  # fields missing from some records are padded
    s = st.series(["loss", "nope"], points=3)
    assert s["n"] == 8 and len(s["step"]) == 3 and set(s["fields"]) == {"loss"}
    assert s["step"][-1] == 8


def test_status_and_token(tmp_path):
    (tmp_path / "run.conf").write_text('STEPS=100\nPARTS=1,3-10\nMODEL_FLAGS="--d_model 8"\n')
    write(tmp_path / "model.msgpack.metrics.jsonl", [train(s, 1.0) for s in range(1, 11)]
          + [{"type": "eval", "step": 10, "ce": 2.0}])
    write(tmp_path / "model.msgpack.samples.jsonl", [{"step": 10, "samples": [{"prompt": "a", "text": "b"}]}])
    st = RunState(str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st, "secret"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/status?token=secret") as r:
            s = json.load(r)
        assert s["step"] == 10 and s["total_steps"] == 100 and s["evals"][0]["ce"] == 2.0
        assert s["conf"]["MODEL_FLAGS"] == "--d_model 8" and abs(s["eta_s"] - 9.0) < 1e-6
        with urllib.request.urlopen(base + "/api/samples?token=secret") as r:
            assert json.load(r)[0]["samples"][0]["text"] == "b"
        with urllib.request.urlopen(base + "/?token=secret") as r:
            assert b"<canvas" in r.read()
        for path in ("/api/status", "/api/status?token=wrong", "/"):
            try:
                urllib.request.urlopen(base + path)
                raise AssertionError("served without the token: " + path)
            except urllib.error.HTTPError as e:
                assert e.code == 403
    finally:
        server.shutdown()


def test_count_parts():
    assert count_parts("1,3-2047") == 2046 and count_parts("") == 0 and count_parts("x") is None
