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
    const one = tk.find(r => r.top_k === 1), trained = tk.find(r => r.top_k === 16);
    return `A fact is <strong>not stored in a single vector</strong>. It is a weighted mix of several. ` +
      `Fetching only the best vector per head (${one.vectors_per_token} in total) gives ${pct(one.accuracy)}, against ${pct(trained.accuracy)} at the trained setting of ${trained.vectors_per_token}. ` +
      `Within each head the strongest vector carries ${pct(mx.top1_weight_mean, 0)} of the weight on average and the top 4 carry ${pct(mx.top4_weight_mean, 0)}. ` +
      `The effective number of vectors mixed is about <strong>${mx.effective_vectors_per_head_mean.toFixed(1)} per head</strong>, ${mx.effective_vectors_per_fact_mean.toFixed(0)} per fact across the 4 heads.`;
  },

  share: (sh, groups) => {
    const g3 = groups.find(g => g.facts_in_vector === 3);
    return `<strong>Yes.</strong> On average each vector is the strongest match for <strong>${sh.facts_per_used_slot_mean.toFixed(1)} different facts</strong> (median ${sh.facts_per_used_slot_pcts[1].toFixed(0)}, max ${sh.facts_per_used_slot_max}). ` +
      `These are really different facts, not one answer reused: ${pct(sh.distinct_answers_fraction_in_shared_slots, 0)} of the facts sharing a vector have distinct answers. ` +
      (g3 ? `For groups of 3 facts with 3 different answers in one vector, all 3 are recalled ${pct(g3.all_correct)} of the time, compared with ${pct(g3.all_correct_random_groups)} for 3 random facts. ` : "") +
      `The table shows this holds as groups get bigger.`;
  },

  triplet: (t, acc) =>
    `<strong>Triplet test.</strong> Fact C shares its strongest vector in one head with fact A, and in another head with fact B. All three have different answers. ` +
    `Over ${num(t.n)} such triplets, all three are correct <strong>${pct(t.all_three_correct)}</strong> of the time, and C alone ${pct(t.c_correct)}. ` +
    `If sharing caused no interference, the expectation from overall accuracy would be ${pct(t.independent_expectation)}.`,

  ablate: (ra, ta, acc) => {
    const half = ra.find(r => r.fraction_deleted === 0.5), all = ra.find(r => r.fraction_deleted === 1);
    const t1 = ta.fact_survives[0], tAll = ta.fact_survives.at(-1);
    return `<strong>Yes, most of the knowledge is in the pool.</strong> Deleting every pool vector drops accuracy from ${pct(acc)} to <strong>${pct(all.accuracy)}</strong>. ` +
      `Knowledge is spread out: deleting a random half of the pool still leaves ${pct(half.accuracy)}. ` +
      `For facts that were correct, deleting the fact's own top vector in each head (${t1.vectors_deleted} vectors) leaves ${pct(t1.accuracy)} of them correct. Deleting all ${tAll.vectors_deleted} vectors it fetches leaves ${pct(tAll.accuracy)}.`;
  },

  neigh: (ta) =>
    `When those top vectors are deleted, <strong>other facts that share them</strong> drop from ${pct(ta.neighbours_sharing_deleted_top1.before)} to ${pct(ta.neighbours_sharing_deleted_top1.after)} correct (${num(ta.neighbours_sharing_deleted_top1.n)} facts). ` +
    `Random unrelated facts go from ${pct(ta.random_control_facts.before)} to ${pct(ta.random_control_facts.after)}. The facts that share a vector really do use it, and deleting it hurts only them.`,

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

const SETUP = [
  `<strong>Data.</strong> Synthetic facts <code>entity, relation → attribute</code>. Entities are multi-token names, so a fact can't hide in a single token embedding. 256 possible answers, so chance is 0.4%. The model sees 10 facts per sequence and is scored only on the answer token.`,
  `<strong>Main model.</strong> 2-layer transformer, width 128. Layer 1 reads from one pool of 4,096 vectors × 128 dims (64 × 64 product keys). 4 router heads × top-16 = 64 vectors per token. 4,000 training steps, batch 64.`,
  `<strong>Baselines.</strong> The same backbone without a pool, and a dense model whose feed-forward layers are 9× wider so its total parameter count matches the pool model. Same data and steps.`,
  `<strong>Capacity sweep.</strong> The pool is fixed at 1,024 vectors, and the number of facts goes from 4,096 to 32,768 (4 to 32 facts per vector). Each point has a backbone-only run with the same step budget.`,
  `<strong>Measurements.</strong> Every test uses the clean router with no training noise. "Strongest vector" means the top-1 vector of a router head. "Deleting" a vector sets its stored values to zero while the router still selects it.`,
  `<strong>Limits.</strong> One seed per setting. CPU training with small fixed step budgets. Synthetic facts, not natural language. Treat differences of a point or two as noise.`,
];
