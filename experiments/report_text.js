const ANSWERS = {
  acc: ({ acc, dm, ds, per, conf }) =>
    `<strong>Yes.</strong> The memory-pool model gets <strong>${pct(acc)}</strong> of the ${num(A.num_facts)} facts right, and ${pct(conf)} are right with more than 90% confidence. ` +
    `The same backbone without a pool reaches ${pct(ds.accuracy)}. A dense model with about the same total parameter count (the pool's parameters moved into a wider feed-forward layer) reaches ${pct(dm.accuracy)}. ` +
    `Accuracy is even across the four relations (${per.map(v => pct(v)).join(", ")}).`,

  cap: (C) => {
    const rows = C.filter(r => r.memory_accuracy != null);
    const best = rows.reduce((a, r) => (r.memory_facts_stored > a.memory_facts_stored ? r : a), rows[0]);
    const dBest = C.filter(r => r.dense_accuracy != null).reduce((a, r) => (r.dense_facts_stored > a.dense_facts_stored ? r : a));
    const pool = best.pool_size;
    return `A pool of <strong>${num(pool)} vectors</strong> stored up to <strong>${num(best.memory_facts_stored)} facts</strong> (${(best.memory_facts_stored / pool).toFixed(1)} facts per vector, at ${num(best.facts)} facts to learn). ` +
      `The backbone alone topped out at ${num(dBest.dense_facts_stored)} facts. ` +
      rows.map(r => `${num(r.facts)} facts → ${pct(r.memory_accuracy)}`).join(", ") + ". " +
      `Each run had a fixed training budget, so the largest settings may still be under-trained rather than full.`;
  },

  merge: (tk, mx) => {
    const one = tk.find(r => r.top_k === 1), two = tk.find(r => r.top_k === 2), four = tk.find(r => r.top_k === 4);
    const trained = tk.find(r => r.top_k === 16), most = tk.at(-1);
    return `Each answer is <strong>a blend of many vectors</strong>, weighted almost equally. ` +
      `In each head the strongest vector gets ${pct(mx.top1_weight_mean, 1)} of the weight (an even split of 16 would be 6.25%), so all ${trained.top_k} fetched vectors count about the same: ` +
      `about ${mx.effective_vectors_per_head_mean.toFixed(0)} per head, ${mx.effective_vectors_per_fact_mean.toFixed(0)} per fact. ` +
      `Yet most of each fact can be recovered from its best few vectors: keeping only the best 1 per head (${one.vectors_per_token} vectors) still gives <strong>${pct(one.accuracy)}</strong>, ` +
      `the best 2 give ${pct(two.accuracy)}, and the best 4 give ${pct(four.accuracy)}, against ${pct(trained.accuracy)} at the trained setting. ` +
      `Fetching more than it was trained with does no harm (${most.vectors_per_token} vectors: ${pct(most.accuracy)}).`;
  },

  share: (sh, groups) => {
    const g3 = groups.find(g => g.facts_in_vector === 3), gmax = groups.at(-1);
    return `<strong>Yes.</strong> Each vector is the strongest match for <strong>${sh.facts_per_used_slot_mean.toFixed(1)} facts on average</strong> (median ${sh.facts_per_used_slot_pcts[1].toFixed(0)}, up to ${sh.facts_per_used_slot_max}). ` +
      `They are genuinely different facts: ${pct(sh.distinct_answers_fraction_in_shared_slots, 0)} of the facts sharing a vector have different answers. ` +
      (g3 ? `When 3 facts with 3 different answers share a vector, all 3 are right ${pct(g3.all_correct)} of the time, the same as for 3 random facts (${pct(g3.all_correct_random_groups)}). ` : "") +
      `Accuracy doesn't drop as vectors get more crowded: ${A.accuracy_by_slot_load.map(b => `${b.facts_per_vector.replace('-+', '+')} facts per vector → ${pct(b.accuracy)}`).join(", ")}.` +
      (gmax ? ` Even with ${gmax.facts_in_vector} different facts in one vector, all are recalled ${pct(gmax.all_correct)} of the time.` : "");
  },

  triplet: (t, acc) =>
    `<strong>Triplet test.</strong> Fact C shares its strongest vector in one head with fact A, and in another head with fact B. All three have different answers. ` +
    `Across ${num(t.n)} such triplets, all three are right <strong>${pct(t.all_three_correct)}</strong> of the time, and C alone ${pct(t.c_correct)}. ` +
    `If sharing caused no interference at all, we'd expect ${pct(t.independent_expectation)}. The gap is ${((t.independent_expectation - t.all_three_correct) * 100).toFixed(2)} points: at most a very small interference.`,

  ablate: (ra, ta, acc) => {
    const q = ra.find(r => r.fraction_deleted === 0.25), half = ra.find(r => r.fraction_deleted === 0.5);
    const tq = ra.find(r => r.fraction_deleted === 0.75), all = ra.find(r => r.fraction_deleted === 1);
    const t1 = ta.fact_survives[0], t4 = ta.fact_survives[2], tAll = ta.fact_survives.at(-1);
    return `<strong>Partly.</strong> With every pool vector deleted, accuracy falls from ${pct(acc)} to <strong>${pct(all.accuracy)}</strong>. ` +
      `So the small backbone memorized about half the facts by itself, and the pool holds the rest plus backup copies. ` +
      `Storage is spread out and redundant: deleting a random ${pct(q.fraction_deleted, 0)} of the pool leaves ${pct(q.accuracy)}, half leaves ${pct(half.accuracy)}, and three quarters leaves ${pct(tq.accuracy)}. ` +
      `No single vector is critical. For facts that were correct, deleting each one's top ${t1.vectors_deleted} vectors leaves ${pct(t1.accuracy)} still correct, and its top ${t4.vectors_deleted} leaves ${pct(t4.accuracy)}. ` +
      `Only deleting all ${tAll.vectors_deleted} vectors a fact fetches brings it down to ${pct(tAll.accuracy)}, about the no-pool level.`;
  },

  neigh: (ta) =>
    `Deleting a fact's top vectors also doesn't hurt <strong>other facts that share those vectors</strong>: they go from ${pct(ta.neighbours_sharing_deleted_top1.before)} to ${pct(ta.neighbours_sharing_deleted_top1.after)} correct (${num(ta.neighbours_sharing_deleted_top1.n)} facts), about the same as unrelated facts (${pct(ta.random_control_facts.before)} → ${pct(ta.random_control_facts.after)}). ` +
    `Each fact is written across many vectors, so losing a few is covered by the rest.`,

  ret: (R) => {
    const last = (k) => R.curves[k]?.at(-1), atA = (k) => R.curves[k]?.filter(p => p.step <= R.steps_A).at(-1);
    const parts = [
      ["memory/full", "the pool model (training everything)"],
      ["memory/pool_values_only", "the pool model (training only pool vectors)"],
      ["dense_matched/full", "the dense model"],
    ].filter(([k]) => R.curves[k]).map(([k, n]) =>
      `${n} keeps <strong>${pct(last(k).acc_A)}</strong> of the old facts (was ${pct(atA(k).acc_A)}) and learns ${pct(last(k).acc_B)} of the new ones`);
    return `Each model first learned ${num(R.facts_A)} facts (set A), then trained only on ${num(R.facts_B)} new facts (set B) for ${num(R.steps_B)} steps. After that, ` +
      parts.join("; ") + `.`;
  },
};

ANSWERS.gen = (G) => {
  const m = G.models.memory.final, d = G.models.dense_matched?.final;
  const row = (k) => `${pct(m[k].accuracy)}` + (d ? ` (dense ${pct(d[k].accuracy)})` : "");
  return `Memorized facts: ${row("memorised")}. ` +
    `<strong>Paraphrase transfer</strong> (a fact learned only as "X r → Y", asked as "X r′ → ?"): ${row("paraphrase_transfer")}. ` +
    `<strong>Rule on entities never seen</strong>: ${row("rule_unseen_entities")}. ` +
    `Rule-breaking exceptions remembered: ${row("exceptions_memorised")}. ` +
    `On facts it can't know (unseen entities, random relations) accuracy is ${row("unknowable")} against 0.4% chance, and the model is still over 50% confident but wrong ${pct(m.unknowable.confident_wrong)} of the time` +
    (d ? ` (dense ${pct(d.unknowable.confident_wrong)})` : "") + `.`;
};

const SETUP = [
  `<strong>Data.</strong> Synthetic facts <code>entity, relation → attribute</code>. Entities are multi-token names, so a fact can't hide in a single token embedding. 256 possible answers, so chance is 0.4%. The model sees 10 facts per sequence and is scored only on the answer token.`,
  `<strong>Main model.</strong> 2-layer transformer, width 128. Layer 1 reads from one pool of 4,096 vectors × 128 dims (64 × 64 product keys). 4 router heads × top-16 = 64 vectors per token. 4,000 training steps, batch 64.`,
  `<strong>Baselines.</strong> The same backbone without a pool, and a dense model whose feed-forward layers are 9× wider so its total parameter count matches the pool model. Same data and steps.`,
  `<strong>Capacity sweep.</strong> The pool is fixed at 1,024 vectors, and the number of facts goes from 4,096 to 32,768 (4 to 32 facts per vector). Each point has a backbone-only run with the same step budget.`,
  `<strong>Measurements.</strong> Every test uses the clean router with no training noise. "Strongest vector" means the top-1 vector of a router head. "Deleting" a vector sets its stored values to zero while the router still selects it.`,
  `<strong>Generalization test.</strong> 2,048 entities: 80% used in training, 20% held out entirely. Relation 0 follows a rule set by the first two name tokens, and 10% of entities break it. Relations 1–3 are random. Every relation also has a paraphrase token; only half of the training entities are ever shown with it.`,
  `<strong>Limits.</strong> One seed per setting. CPU training with small fixed step budgets. Synthetic facts, not natural language. Treat differences of a point or two as noise.`,
];
