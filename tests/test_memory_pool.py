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


def test_top_k_by_max_matches_lax_top_k():
    from memory_pool_model.memory import _top_k_by_max
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 100))
    x = x.at[:, 10].set(x[:, 3])  # ties
    for k in (1, 4, 16):
        v, i = _top_k_by_max(x, k)
        v2, i2 = jax.lax.top_k(x, k)
        np.testing.assert_array_equal(i, i2)
        np.testing.assert_array_equal(v, v2)


def test_pallas_top_k_matches_lax_top_k():
    from memory_pool_model import topk_pallas
    x = jax.random.normal(jax.random.PRNGKey(0), (40, 256))  # rows not a multiple of the block
    x = x.at[3, :].set(0.0).at[3, 7].set(1.0)  # ties: lowest index first
    v, i = topk_pallas.top_k(x, 5, interpret=True)
    v_ref, i_ref = jax.lax.top_k(x, 5)
    np.testing.assert_array_equal(i, i_ref)
    np.testing.assert_array_equal(v, v_ref)
    assert topk_pallas.supported(512, 16) and not topk_pallas.supported(16, 4)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")
def test_fused_router_matches_xla():
    from memory_pool_model import router_pallas as rp
    G, M, n, d, k = 4, 300, 128, 16, 4
    kq, kk, kn, kw1, kw2 = jax.random.split(jax.random.PRNGKey(0), 5)
    nrm = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    q, keys = nrm(jax.random.normal(kq, (G, M, d))), nrm(jax.random.normal(kk, (G, n, d)))
    t, noise = jnp.float32(12.0), 0.1 * jax.random.gumbel(kn, (G, M, n))
    w1, w2 = jax.random.normal(kw1, (G, M, k)), jax.random.normal(kw2, (G, n))

    def ref(q, keys, t):
        cos = jnp.einsum("gmd,gnd->gmn", q, keys)
        scores = cos * t
        sel_vals, idx = jax.lax.top_k(jax.lax.stop_gradient(scores + noise), k)
        kth = jax.lax.top_k(jax.lax.stop_gradient(scores), k)[0][..., -1:]
        sub = jnp.take_along_axis(scores, idx, -1)
        return idx, sub, sel_vals, (scores >= kth).sum(1), jax.nn.softmax(cos * jax.lax.stop_gradient(t), -1).sum(1)

    with jax.default_matmul_precision("highest"):
        a = rp.fused_router(q, keys, t, noise, k, True)
        b = ref(q, keys, t)
        np.testing.assert_array_equal(a[0], b[0])
        for x, y in zip(a[1:5], b[1:5]):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)
        loss = lambda o: jnp.sum(o[1] * w1) + jnp.sum(o[4] * w2)
        ga = jax.grad(lambda *a: loss(rp.fused_router(*a, noise, k, True)), argnums=(0, 1, 2))(q, keys, t)
        gb = jax.grad(lambda *a: loss(ref(*a)), argnums=(0, 1, 2))(q, keys, t)
        for x, y in zip(ga, gb):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-5)


def test_onehot_take_matches_gather_and_its_gradient():
    from memory_pool_model.memory import _onehot_take, _onehot_take_diff
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 2, 32))
    idx = jax.random.randint(jax.random.PRNGKey(1), (3, 4, 2, 5), 0, 32)
    g = jax.random.normal(jax.random.PRNGKey(2), idx.shape)
    ref, ref_vjp = jax.vjp(lambda x: jnp.take_along_axis(x, idx, -1), x)
    out, vjp = jax.vjp(lambda x: _onehot_take_diff(x, idx, 32), x)
    np.testing.assert_allclose(out, ref, rtol=1e-6)
    np.testing.assert_allclose(vjp(g)[0], ref_vjp(g)[0], rtol=1e-6, atol=1e-6)  # repeated indices add up
    ints = jax.random.randint(jax.random.PRNGKey(3), (3, 4, 16), 0, 100)
    np.testing.assert_array_equal(_onehot_take(ints, idx[:, :, 0] % 16), jnp.take_along_axis(ints, idx[:, :, 0] % 16, -1))


def test_count_ids_matches_scatter_counts():
    from memory_pool_model.memory import _count_ids
    ids = jax.random.randint(jax.random.PRNGKey(0), (50, 3, 4), 0, 16)
    ref = jnp.zeros((3, 16)).at[jnp.arange(3)[None, :, None], ids].add(1.0)
    np.testing.assert_array_equal(_count_ids(ids, 16), ref)


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


def test_temperature_floor_and_balance_gradient():
    def temp_grads(pool_kw, log_t):
        pool = MemoryPool(n_sub_keys=N_SUB, heads=HEADS, d_key=D_KEY, d_value=D_VALUE, top_k=K, **pool_kw)
        q = jax.random.normal(jax.random.PRNGKey(0), (3, 5, HEADS, D_KEY))
        params = pool.init(jax.random.PRNGKey(1), q)
        params = jax.tree_util.tree_map(lambda x: x, params)
        params["params"]["log_temperature"] = jnp.asarray(log_t)

        def f(p, what):
            out, aux = pool.apply(p, q)
            return aux["balance_loss"] if what == "balance" else jnp.sum(out**2)

        g = lambda what: float(jax.grad(f)(params, what)["params"]["log_temperature"])
        return g("balance"), g("read"), float(pool.apply(params, q)[1]["temperature"])

    # the floor clamps the value but still passes gradient (no dead temperature)
    _, g_read, t = temp_grads({"min_temperature": 5.0}, jnp.log(2.0))
    # relative: TPU's exp is approximate (5.0 comes back as 5.000105)
    assert abs(t - 5.0) < 5e-5 * 5.0 and g_read != 0.0
    # the balance loss can't lower its value by flattening the router
    g_bal, _, _ = temp_grads({"balance_temperature_grad": True}, jnp.log(10.0))
    assert g_bal != 0.0
    g_bal, g_read, _ = temp_grads({"balance_temperature_grad": False}, jnp.log(10.0))
    assert g_bal == 0.0 and g_read != 0.0


def test_state_types_are_stable_across_steps():
    """Any dtype / weak-type / shape change in the state makes the next
    jitted step recompile (it cost ~10 s per change on a T4)."""
    ds = FactDataset(num_entities=16, num_relations=2, num_attributes=8, name_len=2)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=16, n_layers=2, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=8, d_value=8, top_k=2)
    tr = Trainer(mcfg, TrainConfig(revive_threshold=1.0))
    state = tr.init(jax.random.PRNGKey(0))
    sig = lambda s: [(x.dtype, getattr(x, "weak_type", False), x.shape) for x in jax.tree_util.tree_leaves(s)]
    s1, _, queries = tr.train_step(state, ds.sample(np.random.default_rng(0), 4), jax.random.PRNGKey(1))
    s2, _ = tr.revive(s1, queries, jax.random.PRNGKey(2))
    assert sig(state) == sig(s1) == sig(s2)


def test_query_scale_sharpens_reads_without_changing_picks():
    pool = MemoryPool(n_sub_keys=N_SUB, heads=HEADS, d_key=D_KEY, d_value=D_VALUE, top_k=K, query_scale=True)
    q = jax.random.normal(jax.random.PRNGKey(0), (3, 5, HEADS, D_KEY))
    params = pool.init(jax.random.PRNGKey(1), q)
    _, a = pool.apply(params, q)
    _, b = pool.apply(params, q * 10.0)
    np.testing.assert_array_equal(a["slots"], b["slots"])  # same vectors (cosine order)
    assert float(b["top1_weight"]) > float(a["top1_weight"])  # but a sharper mix


def test_pool_values_only_changes_nothing_else():
    ds = FactDataset(num_entities=16, num_relations=2, num_attributes=8, name_len=2)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=16, n_layers=2, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=8, d_value=8, top_k=2)
    tr = Trainer(mcfg, TrainConfig(pool_values_only=True, warmup_steps=1))
    s0 = tr.init(jax.random.PRNGKey(0))
    s1 = s0
    for i in range(2):  # the learning rate is 0 at step 0 (warmup)
        s1, _, _ = tr.train_step(s1, ds.sample(np.random.default_rng(i), 4), jax.random.PRNGKey(i))
    flat0 = jax.tree_util.tree_flatten_with_path(s0.params)[0]
    for (path, a), b in zip(flat0, jax.tree_util.tree_leaves(s1.params)):
        if _path_is(path, "pool", "values"):
            assert not np.allclose(a, b)
        else:
            np.testing.assert_array_equal(a, b)


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
        # noise-free eval usage, not training-time slot_spread_ema: that one
        # mostly measures the routing noise (0.53 at noise 1.0, 0.41 at 0.1
        # here, with eval coverage 0.59 vs 0.61). A collapse gives ~1/N.
        assert ev["pool_coverage"] > 0.5
        assert ev["pool_active"] > 0.3 and ev["pool_spread"] > 0.15


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
    # the sort-free dense version gives the same table
    from memory_pool_model.train import fetched_row_grads_dense
    dense_rows = fetched_row_grads_dense(aux, g_probe, [1], trainer.mcfg.pool_size)
    np.testing.assert_allclose(dense_rows, g_full["pool"]["values"], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("pool_optimizer", ["adam", "rowwise_adagrad"])
def test_dense_and_unique_row_grads_train_the_same(pool_optimizer):
    ds = FactDataset(num_entities=64, num_relations=2, num_attributes=16, name_alphabet=8, name_len=2, facts_per_seq=4)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=32, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=16, d_value=32, top_k=4)
    states, metrics = [], []
    for mode in ("dense", "unique"):
        tr = Trainer(mcfg, TrainConfig(batch_size=16, warmup_steps=2, pool_row_grads=mode,
                                       pool_optimizer=pool_optimizer))
        assert tr.dense_row_grads == (mode == "dense")
        s = tr.init(jax.random.PRNGKey(0))
        rng = np.random.default_rng(0)
        for i in range(6):
            s, m, _ = tr.train_step(s, ds.sample(rng, 16), jax.random.PRNGKey(i))
        states.append(s)
        metrics.append(m)
    assert int(metrics[0]["rows_updated"]) == int(metrics[1]["rows_updated"])
    flat = jax.tree_util.tree_flatten_with_path(states[0])[0]
    for (path, a), b in zip(flat, jax.tree_util.tree_leaves(states[1])):
        # attention key biases have a true gradient of exactly 0 (softmax
        # ignores a shift shared by all keys): they only see rounding noise,
        # which Adam scales up to full-size steps, so they differ between
        # any two summation orders
        if "['key']['bias']" in jax.tree_util.keystr(path):
            continue
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6)


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


@pytest.mark.parametrize("row_grads", ["unique", "dense"])
def test_data_parallel_matches_single_device(row_grads):
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
        tcfg = TrainConfig(steps=10, batch_size=16, warmup_steps=2, pool_row_grads="ROW_GRADS")
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
    """).replace("ROW_GRADS", row_grads)
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
    # full-precision matmuls: TPU's default bf16 passes make two differently
    # compiled steps differ by ~1e-3 in the loss
    with jax.default_matmul_precision("highest"):
        _host_pool_training_matches_device_pool(pool_optimizer)


def _host_pool_training_matches_device_pool(pool_optimizer):
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
    # full f32 matmuls: TPUs default to bf16 passes, and the cached and full
    # paths then round differently (up to 0.03 on logits of ~3)
    with jax.default_matmul_precision("highest"):
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


def _tiny(**tkw):
    ds = FactDataset(num_entities=16, num_relations=2, num_attributes=8, name_len=2)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=16, n_layers=2, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=8, d_value=8, top_k=2)
    return ds, Trainer(mcfg, TrainConfig(warmup_steps=1, **tkw))


def test_stored_temperature_cannot_drift_past_its_bounds():
    """The forward clamp is straight-through; the trainer projects the
    stored value back so it can't wander below the floor unseen."""
    ds, tr = _tiny()
    s = tr.init(jax.random.PRNGKey(0))
    below = {**s.params, "pool": {**s.params["pool"], "log_temperature": jnp.log(jnp.float32(0.5))}}
    s = s.replace(params=below)  # e.g. an old checkpoint that had drifted
    for i in range(3):
        s, _, _ = tr.train_step(s, ds.sample(np.random.default_rng(i), 4), jax.random.PRNGKey(i))
        lt = float(s.params["pool"]["log_temperature"])
        assert np.log(tr.mcfg.min_temperature) - 1e-6 <= lt <= np.log(tr.mcfg.max_temperature) + 1e-6


def test_pool_values_only_dense_updates():
    ds, tr = _tiny(pool_values_only=True, sparse_pool_updates=False)
    s0 = tr.init(jax.random.PRNGKey(0))
    s1 = s0
    for i in range(2):
        s1, _, _ = tr.train_step(s1, ds.sample(np.random.default_rng(i), 4), jax.random.PRNGKey(i))
    flat0 = jax.tree_util.tree_flatten_with_path(s0.params)[0]
    for (path, a), b in zip(flat0, jax.tree_util.tree_leaves(s1.params)):
        if _path_is(path, "pool", "values"):
            assert not np.allclose(a, b)
        else:
            np.testing.assert_array_equal(a, b)


def test_retention_phase_b_setup_runs_with_sparse_updates():
    """experiments.knowledge_tests.retention re-inits the optimizer for phase B."""
    import optax
    from experiments.knowledge_tests import freeze_except_pool_values
    ds, tr_a = _tiny()
    state = tr_a.init(jax.random.PRNGKey(0))
    state, _, _ = tr_a.train_step(state, ds.sample(np.random.default_rng(0), 4), jax.random.PRNGKey(0))
    _, tr_b = _tiny()
    tr_b.optimizer = optax.chain(tr_b.optimizer, freeze_except_pool_values())
    fresh = tr_b.init(jax.random.PRNGKey(0))
    st = state.replace(opt_state=fresh.opt_state, pool_m=fresh.pool_m, pool_v=fresh.pool_v)
    for i in range(2):
        st, _, _ = tr_b.train_step(st, ds.sample(np.random.default_rng(i), 4), jax.random.PRNGKey(i))
    assert not np.allclose(st.params["pool"]["values"], state.params["pool"]["values"])
    np.testing.assert_array_equal(st.params["embed"]["embedding"], state.params["embed"]["embedding"])


def test_revived_usage_is_still_a_distribution():
    H, C, n, d = 2, 2, 8, 4
    keys = _l2_normalize(jax.random.normal(jax.random.PRNGKey(0), (H, C, n, d)))
    usage = jax.random.dirichlet(jax.random.PRNGKey(1), jnp.full((n,), 0.3), (H, C))
    queries = _l2_normalize(jax.random.normal(jax.random.PRNGKey(2), (32, H, C, d)))
    _, new_usage, dead = revive_dead_keys(keys, usage, queries, jax.random.PRNGKey(3), 0.5)
    assert int(dead.sum()) > 0
    np.testing.assert_allclose(new_usage.sum(-1), 1.0, rtol=1e-5)


def test_shuffle_reads_keeps_routing_but_changes_what_is_read():
    pool, params, q = make_pool()
    out, aux = pool.apply(params, q)
    out_s, aux_s = pool.apply(params, q, shuffle_reads=True)
    np.testing.assert_array_equal(aux["slots"], aux_s["slots"])
    assert not np.allclose(out, out_s)
    # it reads exactly the shifted rows
    v = params["params"]["values"]
    N = N_SUB**2
    read = (aux["slots"] + N // 2 + 1) % N
    manual = jnp.einsum("...hk,...hkd->...d", aux["weights"], v[read])
    np.testing.assert_allclose(out_s, manual, rtol=1e-5, atol=1e-6)


def test_eval_reports_shuffled_pool_accuracy():
    ds, tr = _tiny()
    s = tr.init(jax.random.PRNGKey(0))
    ev = tr.evaluate(s.params, ds, 8)
    assert {"acc_shuffled_pool", "ce_shuffled_pool"} <= set(ev)


def test_zero_value_init_starts_silent_and_still_learns():
    ds = FactDataset(num_entities=16, num_relations=2, num_attributes=8, name_len=2)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, max_len=ds.seq_len, d_model=16, n_layers=2, n_heads=2,
                       n_sub_keys=8, pool_heads=2, d_key=8, d_value=8, top_k=2, value_init_scale=0.0)
    tr = Trainer(mcfg, TrainConfig(warmup_steps=1))
    s = tr.init(jax.random.PRNGKey(0))
    assert not np.any(s.params["pool"]["values"])
    for i in range(2):  # the learning rate is 0 at step 0 (warmup)
        s, _, _ = tr.train_step(s, ds.sample(np.random.default_rng(i), 4), jax.random.PRNGKey(i))
    assert np.any(s.params["pool"]["values"])


def test_decay_pool_path_option():
    from memory_pool_model.train import make_optimizer
    ds, tr = _tiny()
    params = tr.init(jax.random.PRNGKey(0)).params
    for on in (True, False):
        # zero gradients: whatever moves is moved by weight decay alone
        opt = make_optimizer(TrainConfig(decay_pool_path=on, lr=1.0, warmup_steps=1, weight_decay=0.5), clip=False)
        state = opt.init(params)
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        for _ in range(2):  # step 0 has lr 0 (warmup)
            upd, state = opt.update(zeros, state, params)
        moved = {k for k, v in upd.items() if any(np.any(x) for x in jax.tree_util.tree_leaves(v))}
        assert ("mem_out_1" in moved) == on and ("ffn_in_0" in moved)
        assert "pool" not in moved


def test_pick_agreement_tracks_routing_noise():
    pool, params, q = make_pool(noise=0.0)
    rngs = {"routing": jax.random.PRNGKey(3)}
    assert float(pool.apply(params, q, train=True, rngs=rngs)[1]["pick_agreement"]) == 1.0
    loud, _, _ = make_pool(noise=100.0)
    quiet, _, _ = make_pool(noise=0.01)
    a_loud = float(loud.apply(params, q, train=True, rngs=rngs)[1]["pick_agreement"])
    a_quiet = float(quiet.apply(params, q, train=True, rngs=rngs)[1]["pick_agreement"])
    assert a_loud < a_quiet <= 1.0
