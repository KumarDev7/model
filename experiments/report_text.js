const ANSWERS = {
  acc: ({ acc, dm, ds, per, conf }) =>
    `<strong>It answers correctly, but on this task the pool doesn't beat a dense model of the same size.</strong> ` +
    `The memory-pool model gets <strong>${pct(acc)}</strong> of the ${num(A.num_facts)} facts right (${pct(conf)} with more than 90% confidence). ` +
    `A dense model with about the same parameter count (${num(dm.params)} vs ${num(D.find(r => r.name === "main").params)}) gets <strong>${pct(dm.accuracy)}</strong>, and got there faster. ` +
    `Even the small backbone alone (${num(ds.params)} parameters, no pool) reaches ${pct(ds.accuracy)}. ` +
    `At 16,384 facts this task is too small to separate the designs; the capacity test below pushes harder.`,

  cap: (C) => {
    const top = C.at(-1);
    const rows = C.filter(r => r.memory_accuracy != null);
    return `A pool of <strong>${num(top.pool_size)} vectors</strong> stores every fact up to ${num(16384)} facts (16 per vector): ` +
      rows.map(r => `${num(r.facts)} → ${pct(r.memory_accuracy)}`).join(", ") + `. ` +
      `The backbone alone keeps up until then (${pct(C.find(r => r.facts === 16384).dense_accuracy)} at ${num(16384)}). ` +
      `At ${num(top.facts)} facts both run out of room within this training budget, but the pool model stores <strong>${num(top.memory_facts_stored)}</strong> facts against <strong>${num(top.dense_facts_stored)}</strong> for the backbone alone (+${pct(top.memory_facts_stored / top.dense_facts_stored - 1, 0)}). ` +
      `The backbone-only model is smaller, though, and no same-size dense model was run at this point, so this shows extra capacity, not better capacity per parameter.`;
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
    const f = last("memory/full"), po = last("memory/pool_values_only"), dn = last("dense_matched/full");
    return `<strong>This is where the pool helps most.</strong> Each model first learned ${num(R.facts_A)} facts (set A, 100% correct), then trained only on ${num(R.facts_B)} new facts (set B) for ${num(R.steps_B)} steps. ` +
      `Training every weight wipes out almost all the old facts, with or without a pool: the pool model keeps ${pct(f.acc_A)} and the dense model ${pct(dn.acc_A)}. ` +
      `Training <strong>only the pool vectors</strong> (backbone and router frozen) learns the new facts just as well (${pct(po.acc_B)}) and keeps <strong>${pct(po.acc_A)}</strong> of the old ones. ` +
      `That's much better, but not forgetting-free: adding knowledge safely still needs replay of old facts or reserved empty vectors.`;
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
    (d ? ` (dense ${pct(d.unknowable.confident_wrong)})` : "") + `. ` +
    `So both models generalize: a fact learned in one wording works in another, and the rule carries over to new entities. The pool model is a little weaker on the rule and more often confidently wrong about things it can't know.`;
};

ANSWERS.verdict = () => {
  const dm = D.find(r => r.name === "dense_matched"), top = C.at(-1), po = R.curves["memory/pool_values_only"].at(-1), dn = R.curves["dense_matched/full"].at(-1);
  return `<strong>Bottom line.</strong> The pool trains without collapsing (${pct(A.pool_usage_on_facts.active_10pct_of_fair_share)} of vectors get a fair share), reaches ${pct(A.accuracy.top1)} accuracy, and one vector can hold many different facts without errors. ` +
    `But at this scale a dense model of the same size is just as accurate (${pct(dm.accuracy)}) and learns faster. ` +
    `The pool's clear wins are extra capacity (${num(top.memory_facts_stored)} vs ${num(top.dense_facts_stored)} facts stored at the limit) and adding new facts by training only the pool (${pct(po.acc_A)} of old facts kept vs ${pct(dn.acc_A)}).`;
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
