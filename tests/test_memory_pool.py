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
    for i in range(40):
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


def test_route_through_pool_blocks_backbone_shortcut():
    model, params, tokens = _tiny_model()

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
    assert phase_at(TrainConfig(), 100) == {"route": False, "nopool": False, "freeze": False}


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
