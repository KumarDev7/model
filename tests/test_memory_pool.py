import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memory_pool_model.config import ModelConfig, TrainConfig
from memory_pool_model.data import FactDataset
from memory_pool_model.memory import (
    MemoryPool,
    _l2_normalize,
    key_diversity_loss,
    revive_dead_keys,
)
from memory_pool_model.train import Trainer, _path_is

N_SUB, HEADS, D_KEY, D_VALUE, K = 16, 2, 8, 12, 4


def make_pool(noise=0.0):
    pool = MemoryPool(
        n_sub_keys=N_SUB, heads=HEADS, d_key=D_KEY, d_value=D_VALUE, top_k=K, routing_noise=noise
    )
    q = jax.random.normal(jax.random.PRNGKey(0), (3, 5, HEADS, D_KEY))
    params = pool.init({"params": jax.random.PRNGKey(1), "routing": jax.random.PRNGKey(2)}, q)
    return pool, params, q


def test_shapes_and_weights():
    pool, params, q = make_pool()
    out, aux = pool.apply(params, q)
    assert out.shape == (3, 5, D_VALUE)
    assert aux["slots"].shape == (3, 5, HEADS, K)
    np.testing.assert_allclose(aux["weights"].sum(-1), 1.0, rtol=1e-5)
    assert aux["slot_counts"].sum() == 3 * 5 * HEADS * K
    np.testing.assert_allclose(aux["subkey_counts"].sum(-1), 3 * 5 * K)


def test_product_keys_find_exact_top_k():
    """Product-key search must match brute force over all n_sub**2 slots."""
    pool, params, q = make_pool()
    _, aux = pool.apply(params, q)
    p = params["params"]
    temp = jnp.exp(p["log_temperature"])
    qh = _l2_normalize(q.reshape(-1, HEADS, 2, D_KEY // 2))
    keys = _l2_normalize(p["sub_keys"])
    s = jnp.einsum("mhcd,hcnd->mhcn", qh, keys) * temp
    full = (s[:, :, 0, :, None] + s[:, :, 1, None, :]).reshape(-1, HEADS, N_SUB**2)
    _, brute = jax.lax.top_k(full, K)
    got = aux["slots"].reshape(-1, HEADS, K)
    np.testing.assert_array_equal(np.sort(got, -1), np.sort(brute, -1))


def test_noise_changes_selection_only_in_training():
    pool, params, q = make_pool(noise=5.0)
    rngs = {"routing": jax.random.PRNGKey(3)}
    _, clean = pool.apply(params, q, train=False, rngs=rngs)
    _, noisy = pool.apply(params, q, train=True, rngs=rngs)
    _, clean2 = pool.apply(params, q, train=False, rngs={"routing": jax.random.PRNGKey(4)})
    np.testing.assert_array_equal(clean["slots"], clean2["slots"])
    assert not np.array_equal(clean["slots"], noisy["slots"])
    # annealed to zero: training reads exactly the rows inference reads
    _, off = pool.apply(params, q, train=True, noise_scale=0.0, rngs=rngs)
    np.testing.assert_array_equal(clean["slots"], off["slots"])


def test_noise_anneal_schedule():
    from memory_pool_model.train import noise_scale_at
    assert noise_scale_at(TrainConfig(steps=100), 99) == 1.0  # default: never annealed
    t = TrainConfig(steps=100, noise_anneal_start=0.6, noise_anneal_end=0.9)
    got = [float(noise_scale_at(t, s)) for s in (10, 60, 75, 90, 100)]
    np.testing.assert_allclose(got, [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-6)


def test_balance_loss_detects_collapse():
    pool, params, _ = make_pool()
    spread_q = jax.random.normal(jax.random.PRNGKey(5), (4096, HEADS, D_KEY))
    collapsed_q = jnp.broadcast_to(spread_q[:1], spread_q.shape)
    _, spread = pool.apply(params, spread_q)
    _, collapsed = pool.apply(params, collapsed_q)
    assert spread["balance_loss"] < 1.5
    assert collapsed["balance_loss"] > 2 * spread["balance_loss"]


def test_key_diversity_loss():
    eye = jnp.eye(4)[None, None]  # orthogonal keys
    assert float(key_diversity_loss(eye)) == pytest.approx(0.0, abs=1e-6)
    same = jnp.ones((1, 1, 4, 4))
    assert float(key_diversity_loss(same)) == pytest.approx(1.0, abs=1e-4)


def test_revive_dead_keys():
    H, C, n, d = 2, 2, 8, 4
    keys = _l2_normalize(jax.random.normal(jax.random.PRNGKey(0), (H, C, n, d)))
    usage = jnp.full((H, C, n), 1.0 / n).at[0, 1, 3].set(0.0).at[1, 0, 5].set(0.0)
    queries = _l2_normalize(jax.random.normal(jax.random.PRNGKey(1), (32, H, C, d)))
    new_keys, new_usage, dead = revive_dead_keys(keys, usage, queries, jax.random.PRNGKey(2), 0.1, noise=0.0)
    assert int(dead.sum()) == 2
    np.testing.assert_array_equal(new_keys[~dead], keys[~dead])
    # Revived key sits exactly on one of the queries for its codebook.
    sims = jnp.einsum("md,d->m", queries[:, 0, 1], new_keys[0, 1, 3])
    assert float(sims.max()) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(new_usage, 1.0 / n)


def _tiny_setup(steps=40, **model_kw):
    ds = FactDataset(num_entities=64, num_relations=2, num_attributes=16, name_alphabet=8, name_len=2, facts_per_seq=4)
    mcfg = ModelConfig(
        vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
        n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4, **model_kw,
    )
    tcfg = TrainConfig(steps=steps, batch_size=16, warmup_steps=5, revive_every=10)
    return ds, Trainer(mcfg, tcfg)


def test_revive_clears_optimizer_moments():
    ds, trainer = _tiny_setup()
    state = trainer.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    for i in range(3):
        state, _, queries = trainer.train_step(state, ds.sample(rng, 16), jax.random.PRNGKey(i))
    state = state.replace(subkey_usage=state.subkey_usage.at[0, 0, 2].set(0.0))
    state, n_dead = trainer.revive(state, queries, jax.random.PRNGKey(9))
    assert int(n_dead) >= 1

    def check(path, x):
        if _path_is(path[-2:], "pool", "sub_keys") and x.ndim == 4:
            assert float(jnp.abs(x[0, 0, 2]).sum()) == 0.0
            assert float(jnp.abs(x).sum()) > 0.0
            check.hit += 1

    check.hit = 0
    jax.tree_util.tree_map_with_path(check, state.opt_state)
    assert check.hit >= 2  # Adam mu and nu


@pytest.mark.parametrize("use_memory", [True, False])
def test_training_reduces_loss(use_memory):
    ds, trainer = _tiny_setup(use_memory=use_memory)
    state = trainer.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    first = last = None
    # without the memory-layer FFN, coverage starts lower and climbs with
    # training (31% at step 40, 66% at 160 here; 99.6% on the full run)
    steps = 160 if use_memory else 40
    for i in range(steps):
        state, metrics, queries = trainer.train_step(state, ds.sample(rng, 16), jax.random.PRNGKey(i))
        if use_memory and (i + 1) % 10 == 0:
            state, _ = trainer.revive(state, queries, jax.random.PRNGKey(100 + i))
        first = float(metrics["ce"]) if first is None else first
        last = float(metrics["ce"])
    assert np.isfinite(last) and last < first
    ev = trainer.evaluate(state.params, ds, 16)
    if use_memory:
        assert ev["pool_coverage"] > 0.5
        assert float(metrics["slot_spread_ema"]) > 0.5


def test_eval_batches_cover_every_fact_once():
    ds = FactDataset(num_entities=37, num_relations=3, facts_per_seq=4)
    total = sum(b["mask"].sum() for b in ds.eval_batches(5))
    assert total == ds.num_facts


# ---- options that make the pool carry the knowledge ----
def _tiny_model(**kw):
    from memory_pool_model.model import MemoryPoolLM
    cfg = ModelConfig(vocab_size=40, max_len=12, d_model=32, n_heads=2, n_sub_keys=8,
                      pool_heads=2, d_key=16, d_value=32, top_k=4, **kw)
    model = MemoryPoolLM(cfg)
    tokens = jax.random.randint(jax.random.PRNGKey(0), (2, 12), 0, 40)
    params = model.init(jax.random.PRNGKey(1), tokens)["params"]
    return model, params, tokens


def test_memory_layer_without_ffn():
    _, params, _ = _tiny_model(memory_ffn=False)
    assert "ffn_in_0" in params and "ffn_in_1" not in params  # layer 1 is the memory layer
    _, params, _ = _tiny_model(memory_ffn=True)
    assert "ffn_in_1" in params


def test_route_through_pool_blocks_backbone_shortcut():
    model, params, tokens = _tiny_model(memory_ffn=True)

    def grads(route, pool_off=False):
        f = lambda p: model.apply({"params": p}, tokens, route_through_pool=route, pool_off=pool_off)[0].sum()
        return jax.grad(f)(params)

    g = grads(True)
    # FFN inside the memory layer is cut off from the loss...
    assert float(jnp.abs(g["ffn_in_1"]["kernel"]).sum()) == 0.0
    # ...earlier layers still learn, but only through the router/pool read
    assert float(jnp.abs(g["attn_0"]["query"]["kernel"]).sum()) > 0.0
    assert float(jnp.abs(g["router_1"]["kernel"]).sum()) > 0.0
    # without the pool path there is nothing left to carry gradient to them
    model_np = lambda p: model.apply({"params": p}, tokens, route_through_pool=True, pool_off=True)[0].sum()
    g_np = jax.grad(model_np)(params)
    assert float(jnp.abs(g_np["attn_0"]["query"]["kernel"]).sum()) == 0.0
    # forward values are unchanged by gradient routing
    a = model.apply({"params": params}, tokens)[0]
    b = model.apply({"params": params}, tokens, route_through_pool=True)[0]
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_nopool_kl_trains_and_reports():
    ds, trainer = _tiny_setup()
    trainer.tcfg = TrainConfig(steps=5, batch_size=16, warmup_steps=1, nopool_kl_coef=1.0,
                               route_through_pool=True)
    trainer.train_step = jax.jit(trainer._train_step)
    state = trainer.init(jax.random.PRNGKey(0))
    state, metrics, _ = trainer.train_step(state, ds.sample(np.random.default_rng(0), 16), jax.random.PRNGKey(1))
    assert np.isfinite(float(metrics["nopool_kl"])) and "acc_nopool" in metrics
    assert "acc_nopool" in trainer.evaluate(state.params, ds, 16)


def test_phases_switch_on_at_the_right_step():
    from memory_pool_model.train import phase_at
    t = TrainConfig(route_after_step=10, nopool_true_coef=1.0, nopool_after_step=5, freeze_backbone_after_step=20)
    assert phase_at(t, 5) == {"route": False, "nopool": False, "freeze": False}
    assert phase_at(t, 11) == {"route": True, "nopool": True, "freeze": False}
    assert phase_at(t, 21)["freeze"]
    assert phase_at(TrainConfig(nopool_true_coef=0.0), 100) == {"route": False, "nopool": False, "freeze": False}
    assert phase_at(TrainConfig(), 1)["nopool"]  # the no-pool penalty is on by default


def test_freeze_trains_only_the_pool_path():
    ds, trainer = _tiny_setup()
    trainer.tcfg = TrainConfig(steps=5, batch_size=16, warmup_steps=1, nopool_true_coef=1.0)
    state = trainer.init(jax.random.PRNGKey(0))
    batch = ds.sample(np.random.default_rng(0), 16)
    new = state
    for i in range(3):  # lr is 0 on the first warmup step
        new, metrics, _ = trainer.train_step(new, batch, jax.random.PRNGKey(i), route=False, nopool=True, freeze=True)
    assert "nopool_true_pen" in metrics and np.isfinite(float(metrics["nopool_true_pen"]))
    np.testing.assert_array_equal(new.params["attn_0"]["query"]["kernel"], state.params["attn_0"]["query"]["kernel"])
    np.testing.assert_array_equal(new.params["embed"]["embedding"], state.params["embed"]["embedding"])
    assert not np.array_equal(new.params["pool"]["values"], state.params["pool"]["values"])
    assert not np.array_equal(new.params["router_1"]["kernel"], state.params["router_1"]["kernel"])


# ---- scale: sparse pool updates, resume, data parallel ----
def test_sparse_row_grads_equal_dense_grads():
    from memory_pool_model.train import fetched_row_grads, merge_values, split_values
    ds, trainer = _tiny_setup()
    state = trainer.init(jax.random.PRNGKey(0))
    batch = ds.sample(np.random.default_rng(0), 8)
    rng = jax.random.PRNGKey(3)
    # dense reference
    g_full = jax.grad(lambda p: trainer._loss(p, batch, rng, True)[0])(state.params)
    # sparse: grads for non-pool params + fetched rows only
    values, rest = split_values(state.params)
    B, T = batch["inputs"].shape
    probes = {1: jnp.zeros((B, T, trainer.mcfg.d_value))}
    (_, (_, aux)), (g_rest, g_probe) = jax.value_and_grad(
        lambda r, pr: trainer._loss(merge_values(r, values), batch, rng, True, probes=pr),
        argnums=(0, 1), has_aux=True)(rest, probes)
    uniq, rows = fetched_row_grads(aux, g_probe, [1], trainer.mcfg.pool_size)
    dense_from_sparse = jnp.zeros_like(values).at[uniq].add(rows, mode="drop")
    np.testing.assert_allclose(dense_from_sparse, g_full["pool"]["values"], rtol=1e-4, atol=1e-6)
    _, g_full_rest = split_values(g_full)
    for a, b in zip(jax.tree_util.tree_leaves(g_rest), jax.tree_util.tree_leaves(g_full_rest)):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
    # rows that were never fetched have exactly zero gradient
    untouched = np.setdiff1d(np.arange(trainer.mcfg.pool_size), np.asarray(uniq))
    assert np.all(np.asarray(g_full["pool"]["values"])[untouched] == 0)


def test_sparse_update_changes_only_fetched_rows():
    ds, trainer = _tiny_setup()
    assert trainer.sparse
    state = trainer.init(jax.random.PRNGKey(0))
    batch = ds.sample(np.random.default_rng(0), 2)
    s1, m1, _ = trainer.train_step(state, batch, jax.random.PRNGKey(1))
    s2, m2, _ = trainer.train_step(s1, batch, jax.random.PRNGKey(2))  # lr > 0 from step 2
    changed = np.any(np.asarray(s2.params["pool"]["values"]) != np.asarray(s1.params["pool"]["values"]), axis=1)
    assert 0 < changed.sum() <= int(m2["rows_updated"]) < trainer.mcfg.pool_size


def test_dense_pool_updates_still_train():
    ds = FactDataset(num_entities=64, num_relations=2, num_attributes=16, name_alphabet=8, name_len=2, facts_per_seq=4)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4)
    trainer = Trainer(mcfg, TrainConfig(steps=40, batch_size=16, warmup_steps=5, sparse_pool_updates=False))
    assert not trainer.sparse
    state = trainer.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    losses = []
    for i in range(40):
        state, metrics, _ = trainer.train_step(state, ds.sample(rng, 16), jax.random.PRNGKey(i))
        losses.append(float(metrics["ce"]))
    assert losses[-1] < losses[0]


def test_run_end_to_end_dense_and_sparse(tmp_path):
    from memory_pool_model.train import run
    ds = FactDataset(num_entities=64, num_relations=2, num_attributes=16, name_alphabet=8, name_len=2, facts_per_seq=4)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4)
    for sparse in (True, False):  # run() donates buffers, which the dense path once broke
        tcfg = TrainConfig(steps=6, batch_size=16, warmup_steps=2, revive_every=3, log_every=100,
                           eval_every=100, sparse_pool_updates=sparse)
        _, state, hist = run(mcfg, tcfg, ds, save_path=str(tmp_path / f"m{sparse}"))
        assert int(state.step) == 6 and hist and np.isfinite(hist[-1]["ce"])


def test_resume_is_exact(tmp_path):
    from memory_pool_model.train import run
    ds = FactDataset(num_entities=64, num_relations=2, num_attributes=16, name_alphabet=8, name_len=2, facts_per_seq=4)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4)
    tcfg = TrainConfig(steps=24, batch_size=16, warmup_steps=5, revive_every=5, log_every=100, eval_every=100)
    _, full, _ = run(mcfg, tcfg, ds, save_path=str(tmp_path / "a"))
    run(mcfg, tcfg, ds, save_path=str(tmp_path / "b"), stop_after=11)
    _, resumed, _ = run(mcfg, tcfg, ds, save_path=str(tmp_path / "b"), resume=True)
    for a, b in zip(jax.tree_util.tree_leaves(full), jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_data_parallel_matches_single_device():
    """Runs in a subprocess with 2 virtual CPU devices."""
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import jax, numpy as np
        from jax.sharding import Mesh
        from memory_pool_model.config import ModelConfig, TrainConfig
        from memory_pool_model.data import FactDataset
        from memory_pool_model.train import Trainer
        assert jax.device_count() == 2
        ds = FactDataset(num_entities=64, num_relations=2, num_attributes=16, name_alphabet=8, name_len=2, facts_per_seq=4)
        mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
                           n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4, routing_noise=0.0)
        tcfg = TrainConfig(steps=10, batch_size=16, warmup_steps=2)
        out = []
        for mesh in (None, Mesh(np.array(jax.devices()), ("data",))):
            tr = Trainer(mcfg, tcfg, mesh=mesh)
            st = tr.place_state(tr.init(jax.random.PRNGKey(0)))
            rng = np.random.default_rng(0)
            for i in range(5):
                st, m, _ = tr.train_step(st, tr.place_batch(ds.sample(rng, 16)), jax.random.PRNGKey(i))
            out.append((float(m["loss"]), jax.device_get(st.params)))
        assert abs(out[0][0] - out[1][0]) < 1e-4, (out[0][0], out[1][0])
        # Attention key biases have an exactly-zero true gradient (softmax
        # ignores a shift shared by all keys); Adam turns their float noise
        # into full-size steps, so they legitimately differ. Compare the rest.
        for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(out[0][1])[0],
                                jax.tree_util.tree_leaves(out[1][1])):
            if "'key']['bias'" in jax.tree_util.keystr(path):
                continue
            np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-5)
        print("OK")
    """)
    env = {**__import__("os").environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
           "JAX_PLATFORMS": "cpu"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


# ---- pool off the accelerator (host RAM / SSD) ----
def _host_pair(tmp_path=None, pool_optimizer="adam"):
    ds = FactDataset(num_entities=64, num_relations=2, num_attributes=16, name_alphabet=8, name_len=2, facts_per_seq=4)
    base = dict(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
                n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4)
    tcfg = TrainConfig(steps=24, batch_size=16, warmup_steps=5, revive_every=5, log_every=100, eval_every=100,
                       pool_optimizer=pool_optimizer)
    dev = Trainer(ModelConfig(**base), tcfg)
    host = Trainer(ModelConfig(**base, pool_location="host",
                               pool_dir=str(tmp_path / "pool") if tmp_path else ""), tcfg)
    return ds, tcfg, dev, host


@pytest.mark.parametrize("pool_optimizer", ["adam", "rowwise_adagrad"])
def test_host_pool_training_matches_device_pool(pool_optimizer):
    ds, _, dev, host = _host_pair(pool_optimizer=pool_optimizer)
    s_dev = dev.init(jax.random.PRNGKey(0))
    s_host = host.init(jax.random.PRNGKey(0))
    assert "values" not in s_host.params["pool"]
    host.host.values[:] = np.asarray(s_dev.params["pool"]["values"])  # same starting table
    for name in ("attn_0", "router_1", "embed"):
        chex_equal = jax.tree_util.tree_map(np.array_equal, s_dev.params[name], s_host.params[name])
        assert all(jax.tree_util.tree_leaves(chex_equal))
    rng = np.random.default_rng(0)
    for i in range(6):
        batch = ds.sample(rng, 16)
        s_dev, m_dev, _ = dev.train_step(s_dev, batch, jax.random.PRNGKey(i))
        s_host, m_host, _ = host.train_step(s_host, batch, jax.random.PRNGKey(i))
        assert abs(float(m_dev["loss"]) - float(m_host["loss"])) < 1e-4
    np.testing.assert_allclose(host.host.values, np.asarray(s_dev.params["pool"]["values"]), rtol=1e-4, atol=1e-5)
    ev_dev, ev_host = dev.evaluate(s_dev.params, ds, 16), host.evaluate(s_host.params, ds, 16)
    assert abs(ev_dev["ce"] - ev_host["ce"]) < 1e-3


def test_ssd_pool_trains_and_resumes_exactly(tmp_path):
    from memory_pool_model.train import run
    ds, tcfg, _, _ = _host_pair()
    base = dict(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
                n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4, pool_location="host")
    tr_a, full, _ = run(ModelConfig(**base, pool_dir=str(tmp_path / "pa")), tcfg, ds, save_path=str(tmp_path / "a"))
    assert isinstance(tr_a.host.values, np.memmap) and os.path.exists(tmp_path / "pa" / "values.npy")
    run(ModelConfig(**base, pool_dir=str(tmp_path / "pb")), tcfg, ds, save_path=str(tmp_path / "b"), stop_after=11)
    tr_b, resumed, _ = run(ModelConfig(**base, pool_dir=str(tmp_path / "pb")), tcfg, ds,
                           save_path=str(tmp_path / "b"), resume=True)
    np.testing.assert_array_equal(np.asarray(tr_a.host.values), np.asarray(tr_b.host.values))
    for a, b in zip(jax.tree_util.tree_leaves(full.params), jax.tree_util.tree_leaves(resumed.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("location", ["device", "host"])
def test_kv_cache_decoding_matches_full_forward(location):
    from memory_pool_model.generate import Generator
    from memory_pool_model.host_pool import HostPool
    from memory_pool_model.model import MemoryPoolLM
    cfg = ModelConfig(vocab_size=50, max_len=16, d_model=32, n_heads=2, n_sub_keys=8, pool_heads=2,
                      d_key=16, d_value=32, top_k=4, memory_layers=(0, 1))
    if location == "host":
        hp = HostPool(cfg.pool_size, cfg.d_value, trainable=False, dtype=np.float16)
        cfg = ModelConfig(**{**cfg.__dict__, "pool_location": "host", "host_pool": hp.name})
    toks = jax.random.randint(jax.random.PRNGKey(0), (2, 12), 0, 50)
    model = MemoryPoolLM(cfg)
    params = model.init(jax.random.PRNGKey(1), toks)["params"]
    full, _ = model.apply({"params": params}, toks)
    res = Generator(cfg, params, batch_size=2).generate(np.asarray(toks), n_new=3)
    np.testing.assert_allclose(res["logits"][:, :12], np.asarray(full), atol=1e-4)
    assert res["tokens"].shape == (2, 3)


def test_rowwise_adagrad_state_is_one_float_per_row():
    from memory_pool_model.host_pool import HostPool
    hp = HostPool(1000, 64, optimizer="rowwise_adagrad")
    assert hp.m is None and hp.v is None and hp.acc.shape == (1000,)
    assert hp.nbytes() == 1000 * 64 * 4 + 1000 * 4  # table + one float per row
    before = hp.values.copy()
    hp.update(np.array([3, 7, 1000]), np.ones((3, 64), np.float32), step=1, lr=0.1)  # 1000 = padding
    changed = np.flatnonzero(np.any(hp.values != before, axis=1))
    assert changed.tolist() == [3, 7] and hp.acc[3] == 1.0
